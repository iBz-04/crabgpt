use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn as nn;
use candle_nn::{Module, Optimizer};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;

#[derive(Clone)]
struct Config {
    batch_size: usize,
    block_size: usize,
    max_iters: usize,
    eval_interval: usize,
    eval_iters: usize,
    learning_rate: f64,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    dropout: f64,
    seed: u64,
    max_new_tokens: usize,
    vocab_size: usize,
    temperature: f32,
    top_k: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            batch_size: 32,
            block_size: 256,
            max_iters: 8000,
            eval_interval: 1000,
            eval_iters: 100,
            learning_rate: 3e-4,
            n_embd: 192,
            n_head: 6,
            n_layer: 4,
            dropout: 0.1,
            seed: 1337,
            max_new_tokens: 500,
            vocab_size: 0,
            temperature: 0.9,
            top_k: 40,
        }
    }
}

struct Dataset {
    train: Vec<u32>,
    val: Vec<u32>,
    itos: Vec<char>,
}

fn build_dataset(path: &str) -> Result<Dataset> {
    let text = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let mut vocab: Vec<char> = text.chars().collect();
    vocab.sort_unstable();
    vocab.dedup();

    let stoi: HashMap<char, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, ch)| (*ch, i as u32))
        .collect();
    let itos = vocab;

    let data = encode(&text, &stoi);
    let split = data.len() * 9 / 10;
    Ok(Dataset {
        train: data[..split].to_vec(),
        val: data[split..].to_vec(),
        itos,
    })
}

fn encode(text: &str, stoi: &HashMap<char, u32>) -> Vec<u32> {
    text.chars().map(|ch| stoi[&ch]).collect()
}

fn decode(tokens: &[u32], itos: &[char]) -> String {
    tokens.iter().map(|&i| itos[i as usize]).collect()
}

#[derive(Clone, Copy)]
enum Split {
    Train,
    Val,
}

fn get_batch(
    split: Split,
    data: &Dataset,
    cfg: &Config,
    device: &Device,
    rng: &mut impl Rng,
) -> Result<(Tensor, Tensor)> {
    let source = match split {
        Split::Train => &data.train,
        Split::Val => &data.val,
    };
    if source.len() <= cfg.block_size + 1 {
        bail!("dataset too small for block_size")
    }

    let max_start = source.len() - cfg.block_size - 1;
    let mut x_buf = Vec::with_capacity(cfg.batch_size * cfg.block_size);
    let mut y_buf = Vec::with_capacity(cfg.batch_size * cfg.block_size);

    for _ in 0..cfg.batch_size {
        let start = rng.gen_range(0..max_start);
        x_buf.extend_from_slice(&source[start..start + cfg.block_size]);
        y_buf.extend_from_slice(&source[start + 1..start + 1 + cfg.block_size]);
    }

    let x = Tensor::from_vec(x_buf, (cfg.batch_size, cfg.block_size), device)?;
    let y = Tensor::from_vec(y_buf, (cfg.batch_size, cfg.block_size), device)?;
    Ok((x, y))
}

fn compute_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, t, c) = logits.dims3()?;
    let logits = logits.reshape((b * t, c))?;
    let targets = targets.reshape((b * t,))?.to_dtype(DType::U32)?;
    Ok(nn::loss::cross_entropy(&logits, &targets)?)
}

fn estimate_split(
    model: &GPT,
    split: Split,
    data: &Dataset,
    cfg: &Config,
    device: &Device,
    rng: &mut impl Rng,
) -> Result<f64> {
    let mut total = 0f64;
    for _ in 0..cfg.eval_iters {
        let (xb, yb) = get_batch(split, data, cfg, device, rng)?;
        let logits = model.forward_t(&xb, false)?;
        let loss = compute_loss(&logits, &yb)?;
        total += loss.to_scalar::<f32>()? as f64;
    }
    Ok(total / cfg.eval_iters as f64)
}

fn estimate_loss(
    model: &GPT,
    data: &Dataset,
    cfg: &Config,
    device: &Device,
    rng: &mut impl Rng,
) -> Result<(f64, f64)> {
    let train = estimate_split(model, Split::Train, data, cfg, device, rng)?;
    let val = estimate_split(model, Split::Val, data, cfg, device, rng)?;
    Ok((train, val))
}

fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let idx = Tensor::arange(0u32, t as u32, device)?;
    let i = idx.reshape((t, 1))?.broadcast_as((t, t))?;
    let j = idx.reshape((1, t))?.broadcast_as((t, t))?;
    Ok(i.ge(&j)?.to_dtype(DType::U8)?)
}

struct MultiHeadAttention {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    proj: nn::Linear,
    attn_dropout: nn::Dropout,
    proj_dropout: nn::Dropout,
    n_head: usize,
    head_size: usize,
    scale: f64,
}

impl MultiHeadAttention {
    fn new(cfg: &Config, vb: nn::VarBuilder) -> Result<Self> {
        if cfg.n_embd % cfg.n_head != 0 {
            bail!("n_embd must be divisible by n_head")
        }
        let head_size = cfg.n_embd / cfg.n_head;
        let scale = (head_size as f64).powf(-0.5);
        Ok(Self {
            key: nn::linear_no_bias(cfg.n_embd, cfg.n_embd, vb.pp("key"))?,
            query: nn::linear_no_bias(cfg.n_embd, cfg.n_embd, vb.pp("query"))?,
            value: nn::linear_no_bias(cfg.n_embd, cfg.n_embd, vb.pp("value"))?,
            proj: nn::linear(cfg.n_embd, cfg.n_embd, vb.pp("proj"))?,
            attn_dropout: nn::Dropout::new(cfg.dropout as f32),
            proj_dropout: nn::Dropout::new(cfg.dropout as f32),
            n_head: cfg.n_head,
            head_size,
            scale,
        })
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        let q = q
            .reshape((b, t, self.n_head, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.n_head, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.n_head, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?;
        let wei = (q.matmul(&k_t)? * self.scale)?;

        let mask = causal_mask(t, x.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, self.n_head, t, t))?;
        let neg = Tensor::full(-1e4f32, (b, self.n_head, t, t), x.device())?;
        let wei = mask.where_cond(&wei, &neg)?;

        let wei = nn::ops::softmax(&wei, D::Minus1)?;
        let wei = self.attn_dropout.forward(&wei, train)?;

        let out = wei.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.n_head * self.head_size))?;
        let out = self.proj.forward(&out)?;
        Ok(self.proj_dropout.forward(&out, train)?)
    }
}

struct FeedForward {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dropout: nn::Dropout,
}

impl FeedForward {
    fn new(cfg: &Config, vb: nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: nn::linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("fc1"))?,
            fc2: nn::linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("fc2"))?,
            dropout: nn::Dropout::new(cfg.dropout as f32),
        })
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = gelu(&x)?;
        let x = self.fc2.forward(&x)?;
        Ok(self.dropout.forward(&x, train)?)
    }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    // Approximate GELU used in GPT-2: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    let x3 = x.powf(3.0)?;                         // Tensor
    let x3_scaled = (x3 * 0.044715)?;             // Result -> Tensor after ?
    let inner_pre = (x + x3_scaled)?;             // Result -> Tensor
    let inner = (inner_pre * 0.7978845608028654)?; // Result -> Tensor
    let tanh = inner.tanh()?;                      // Tensor -> Result
    let a = (x * 0.5)?;                            // Tensor
    let b = (tanh + 1.0)?;                         // Tensor
    let out = (a * b)?;                            // Result -> Tensor
    Ok(out)
}

struct Block {
    sa: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
}

impl Block {
    fn new(cfg: &Config, vb: nn::VarBuilder) -> Result<Self> {
        Ok(Self {
            sa: MultiHeadAttention::new(cfg, vb.pp("sa"))?,
            ffwd: FeedForward::new(cfg, vb.pp("ffwd"))?,
            ln1: nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln1"))?,
            ln2: nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln2"))?,
        })
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = x.broadcast_add(&self.sa.forward_t(&self.ln1.forward(x)?, train)?)?;
        let x = x.broadcast_add(&self.ffwd.forward_t(&self.ln2.forward(&x)?, train)?)?;
        Ok(x)
    }
}

struct GPT {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: nn::LayerNorm,
    lm_head: nn::Linear,
}

impl GPT {
    fn new(cfg: &Config, vb: nn::VarBuilder) -> Result<Self> {
        let token_embedding = nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("token_embedding"))?;
        let position_embedding =
            nn::embedding(cfg.block_size, cfg.n_embd, vb.pp("position_embedding"))?;
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(cfg, vb.pp(format!("blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = nn::linear(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
        })
    }

    fn forward_t(&self, idx: &Tensor, train: bool) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let tok_emb = self.token_embedding.forward(idx)?;
        let pos = Tensor::arange(0u32, t as u32, idx.device())?;
        let pos_emb = self.position_embedding.forward(&pos)?;
        let mut x = tok_emb.broadcast_add(&pos_emb)?;
        for block in &self.blocks {
            x = block.forward_t(&x, train)?;
        }
        let x = self.ln_f.forward(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }
}

fn generate(
    model: &GPT,
    cfg: &Config,
    device: &Device,
    data: &Dataset,
    rng: &mut impl Rng,
) -> Result<String> {
    let seed_len = cfg.block_size.min(64);
    let mut idx = if seed_len > 0 && data.train.len() >= seed_len {
        let max_start = data.train.len() - seed_len;
        let start = if max_start > 0 {
            rng.gen_range(0..max_start)
        } else {
            0
        };
        data.train[start..start + seed_len].to_vec()
    } else {
        vec![0u32]
    };
    for _ in 0..cfg.max_new_tokens {
        let start = idx.len().saturating_sub(cfg.block_size);
        let idx_cond = &idx[start..];
        let x = Tensor::from_vec(idx_cond.to_vec(), (1, idx_cond.len()), device)?;
        let logits = model.forward_t(&x, false)?;
        let last = logits
            .narrow(1, idx_cond.len() - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?;
        let mut logits = last.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let temp = cfg.temperature.max(1e-4);
        for v in &mut logits {
            *v /= temp;
        }
        apply_top_k(&mut logits, cfg.top_k);
        let probs = softmax_cpu(&logits);
        let dist = WeightedIndex::new(&probs)?;
        let next_id = dist.sample(rng) as u32;
        idx.push(next_id);
    }
    Ok(decode(&idx, &data.itos))
}

fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(Ordering::Equal)
    });
    let cutoff = logits[idx[top_k - 1]];
    for v in logits {
        if *v < cutoff {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn softmax_cpu(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v - max).exp();
        exps.push(e);
        sum += e;
    }
    if sum == 0.0 {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    for v in &mut exps {
        *v /= sum;
    }
    exps
}

fn save_model(varmap: &nn::VarMap, path: &str) -> Result<()> {
    // Convert all variables to CPU and collect data
    let mut tensors = Vec::new();
    
    for (idx, var) in varmap.all_vars().iter().enumerate() {
        let tensor = var.as_tensor().to_device(&Device::Cpu)?;
        let shape = tensor.shape().dims().to_vec();
        let flat = tensor.reshape((tensor.elem_count(),))?;
        let data = flat.to_vec1::<f32>()?;
        
        // Convert f32 to bytes
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes().to_vec())
            .collect();
        
        tensors.push((idx.to_string(), shape, bytes));
    }
    
    // Save as simple JSON metadata + binary data
    let metadata: Vec<(String, Vec<usize>)> = tensors
        .iter()
        .map(|(name, shape, _)| (name.clone(), shape.clone()))
        .collect();
    
    let metadata_json = serde_json::to_string(&metadata)?;
    fs::write(&format!("{}.meta.json", path), metadata_json)?;
    
    // Save all tensor data concatenated
    let mut all_data = Vec::new();
    for (_, _, bytes) in &tensors {
        all_data.extend_from_slice(bytes);
    }
    fs::write(path, &all_data)?;
    
    println!("✓ Model saved to {} ({} parameters)", path, all_data.len() / 4);
    Ok(())
}

fn main() -> Result<()> {
    let mut cfg = Config::default();
    let data = build_dataset("input.txt")?;
    cfg.vocab_size = data.itos.len();

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    if matches!(device, Device::Metal(_)) {
        device.set_seed(cfg.seed)?;
    }

    let varmap = nn::VarMap::new();
    let vb = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&cfg, vb)?;

    let param_count: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum();
    println!(
        "{:.3} M parameters",
        param_count as f64 / 1_000_000f64
    );

    let mut opt = nn::AdamW::new(
        varmap.all_vars(),
        nn::ParamsAdamW {
            lr: cfg.learning_rate,
            ..Default::default()
        },
    )?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    for iter in 0..cfg.max_iters {
        if iter % cfg.eval_interval == 0 || iter == cfg.max_iters - 1 {
            let (train, val) = estimate_loss(&model, &data, &cfg, &device, &mut rng)?;
            println!(
                "step {iter}: train loss {train:.4}, val loss {val:.4}"
            );
        }

        let (xb, yb) = get_batch(Split::Train, &data, &cfg, &device, &mut rng)?;
        let logits = model.forward_t(&xb, true)?;
        let loss = compute_loss(&logits, &yb)?;
        opt.backward_step(&loss)?;
    }

    let sample = generate(&model, &cfg, &device, &data, &mut rng)?;
    println!("{sample}");

    save_model(&varmap, "model.safetensors")?;

    Ok(())
}

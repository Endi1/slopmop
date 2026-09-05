use crate::jina_model::{BertModel, Config, PositionEmbeddingType};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, VarBuilder};
use hf_hub::HFClientSync;
use tokenizers::{Tokenizer, TruncationParams};

const MODEL_ID: &str = "jina-embeddings-v2-base-code";
const MODEL_OWNER: &str = "jinaai";
const MAX_LENGTH: usize = 8192;

pub struct JinaEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl JinaEmbedder {
    pub fn load() -> Result<Self> {
        let repo = HFClientSync::new()?.model(MODEL_OWNER, MODEL_ID);
        let model_path = repo
            .download_file()
            .filename("model.safetensors")
            .send()
            .context("failed to download Jina model weights")?;
        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .send()
            .context("failed to download Jina tokenizer")?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;

        let config = Config::new(
            tokenizer.get_vocab_size(true),
            768,
            12,
            12,
            3072,
            Activation::Gelu,
            MAX_LENGTH,
            2,
            0.02,
            1e-12,
            0,
            PositionEmbeddingType::Alibi,
        );
        let device = Device::Cpu;
        let weights =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
        let model = BertModel::new(weights, &config)?;

        Ok(Self { model, tokenizer })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Encoding one input at a time avoids padding. Mean pooling can therefore
        // average every returned token while matching the model's attention mask.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let token_ids = Tensor::new(encoding.get_ids(), &self.model.device)?.unsqueeze(0)?;
        let token_embeddings = self.model.forward(&token_ids)?;
        let token_count = token_embeddings.dim(1)?;
        let pooled = (token_embeddings.sum(1)? / token_count as f64)?;
        let normalized = pooled.broadcast_div(&pooled.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        Ok(normalized.get(0)?.to_vec1()?)
    }
}

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{rnn, GRUConfig, LSTMConfig, VarBuilder, RNN};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};

use anyhow::Result;

const ACCURACY: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct Config {
    pub input: usize,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub hidden: usize,
    pub layers: usize,
    pub bidirection: bool,
}

fn repo() -> Result<ApiRepo> {
    let api = Api::new()?;
    let repo_id = "kigichang/test_rnn".to_string();
    let repo = api.repo(Repo::with_revision(
        repo_id,
        RepoType::Model,
        "main".to_string(),
    ));
    Ok(repo)
}

fn load_model(model: &str, device: &Device) -> Result<(Config, VarBuilder<'static>)> {
    let repo = repo()?;

    let filename = repo.get(&format!("{}.pt", model))?;
    let config_file = repo.get(&format!("{}.json", model))?;

    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let vb = VarBuilder::from_pth(filename, DType::F32, device)?;

    Ok((config, vb))
}

fn assert_tensor(a: &Tensor, b: &Tensor, v: f32) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let dim = a.dims().len();
    let mut t = (a - b)?.abs()?;

    for _i in 0..dim {
        t = t.max(D::Minus1)?;
    }

    let t = t.to_scalar::<f32>()?;
    println!("max diff = {}", t);
    assert!(t < v);
    Ok(())
}

fn lstm_config(index: usize, direction: rnn::Direction) -> LSTMConfig {
    let mut config = LSTMConfig::default();
    config.layer_idx = index;
    config.direction = direction;
    config
}

fn gru_config(index: usize, direction: rnn::Direction) -> GRUConfig {
    let mut config = GRUConfig::default();
    config.layer_idx = index;
    config.direction = direction;
    config
}

fn run_lstm(model: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let (config, vb) = load_model(model, device)?;

    let answer = vb.get(
        (config.batch_size, config.sequence_length, config.hidden),
        "output",
    )?;

    let mut layers = Vec::with_capacity(config.layers);

    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden
        };
        let lstm_config = lstm_config(layer_idx, rnn::Direction::Forward);
        let lstm = candle_nn::lstm(input_dim, config.hidden, lstm_config, vb.clone())?;
        layers.push(lstm);
    }

    let mut input = vb.get(
        (config.batch_size, config.sequence_length, config.input),
        "input",
    )?;

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    Ok((input, answer))
}

fn run_bidirectional_lstm(model: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let (config, vb) = load_model(model, device)?;

    let answer = vb.get(
        (config.batch_size, config.sequence_length, config.hidden * 2),
        "output",
    )?;

    let mut layers = Vec::with_capacity(config.layers);

    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden * 2
        };

        let forward_config = lstm_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::lstm(input_dim, config.hidden, forward_config, vb.clone())?;

        let backward_config = lstm_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::lstm(input_dim, config.hidden, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    let mut input = vb.get(
        (config.batch_size, config.sequence_length, config.input),
        "input",
    )?;

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.bidirectional_states_to_tensor(&forward_states, &backward_states)?;
    }
    Ok((input, answer))
}

fn test_lstm() -> Result<()> {
    let (output, answer) = run_lstm("lstm_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn test_nlayer_lstm() -> Result<()> {
    let (output, answer) = run_lstm("lstm_nlayer_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn test_bi_lstm() -> Result<()> {
    let (output, answer) = run_bidirectional_lstm("bi_lstm_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn test_nlayer_bi_lstm() -> Result<()> {
    let (output, answer) = run_bidirectional_lstm("bi_lstm_nlayer_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn run_gru(model: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let (config, vb) = load_model(model, device)?;
    let answer = vb.get(
        (config.batch_size, config.sequence_length, config.hidden),
        "output",
    )?;

    let mut layers = Vec::with_capacity(config.layers);

    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden
        };
        let gru_config = gru_config(layer_idx, rnn::Direction::Forward);
        let gru = candle_nn::gru(input_dim, config.hidden, gru_config, vb.clone())?;
        layers.push(gru);
    }

    let mut input = vb.get(
        (config.batch_size, config.sequence_length, config.input),
        "input",
    )?;

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    Ok((input, answer))
}

fn run_bidirectional_gru(model: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let (config, vb) = load_model(model, device)?;

    let answer = vb.get(
        (config.batch_size, config.sequence_length, config.hidden * 2),
        "output",
    )?;

    let mut layers = Vec::with_capacity(config.layers);
    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden * 2
        };

        let forward_config = gru_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::gru(input_dim, config.hidden, forward_config, vb.clone())?;

        let backward_config = gru_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::gru(input_dim, config.hidden, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    let mut input = vb.get(
        (config.batch_size, config.sequence_length, config.input),
        "input",
    )?;

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.bidirectional_states_to_tensor(&forward_states, &backward_states)?;
    }

    Ok((input, answer))
}

fn test_gru() -> Result<()> {
    let (output, answer) = run_gru("gru_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn test_bi_gru() -> Result<()> {
    let (output, answer) = run_bidirectional_gru("bi_gru_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, 0.000001)
}

fn test_nlayer_gru() -> Result<()> {
    let (output, answer) = run_gru("gru_nlayer_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn test_nlayer_bi_gru() -> Result<()> {
    let (output, answer) = run_bidirectional_gru("bi_gru_nlayer_test", &Device::Cpu)?;
    assert_tensor(&output, &answer, ACCURACY)
}

fn main() -> Result<()> {
    print!("test lstm: ");
    test_lstm()?;

    print!("test gru: ");
    test_gru()?;

    print!("test bi lstm: ");
    test_bi_lstm()?;

    print!("test_bi_gru: ");
    test_bi_gru()?;

    print!("test_nlayer_lstm: ");
    test_nlayer_lstm()?;

    print!("test_nlayer_gru:");
    test_nlayer_gru()?;

    print!("test_nlayer_bi_lstm: ");
    test_nlayer_bi_lstm()?;

    print!("test_nlayer_bi_gru: ");
    test_nlayer_bi_gru()?;

    Ok(())
}

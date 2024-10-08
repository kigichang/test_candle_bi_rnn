use candle_core::{DType, Device, Tensor, D};
use candle_nn::{rnn, GRUConfig, LSTMConfig, VarBuilder, RNN};

use anyhow::Result;

static IN_DIM: usize = 10;
static HIDDEN_DIM: usize = 20;
static BATCH_SIZE: usize = 5;
static SEQ_LEN: usize = 3;

struct Args {
    input_dim: usize,

    hidden_dim: usize,

    layers: usize,

    batch_size: usize,

    seq_len: usize,

    pt: &'static str,
}

impl Args {
    pub fn load_pt(&self) -> Result<VarBuilder<'static>> {
        Ok(VarBuilder::from_pth(self.pt, DType::F32, &Device::Cpu)?)
    }
}

fn assert_tensor(a: &Tensor, b: &Tensor, dim: usize, v: f32) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let mut t = (a - b)?.abs()?;

    for _i in 0..dim {
        t = t.max(D::Minus1)?;
    }

    let t = t.to_scalar::<f32>()?;
    println!("max diff: {}", t);
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

fn run_lstm(args: Args) -> Result<(Tensor, Tensor)> {
    let vb = args.load_pt()?;
    let answer = vb.get((args.batch_size, args.seq_len, args.hidden_dim), "output")?;

    let mut layers = Vec::with_capacity(args.layers);

    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim
        };
        let config = lstm_config(layer_idx, rnn::Direction::Forward);
        let lstm = candle_nn::lstm(input_dim, args.hidden_dim, config, vb.clone())?;
        layers.push(lstm);
    }

    let mut input = vb.get((args.batch_size, args.seq_len, args.input_dim), "input")?;

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    Ok((input, answer))
}

fn run_bidirectional_lstm(args: Args) -> Result<(Tensor, Tensor)> {
    let vb = args.load_pt()?;
    let answer = vb.get(
        (args.batch_size, args.seq_len, args.hidden_dim * 2),
        "output",
    )?;

    let mut layers = Vec::with_capacity(args.layers);

    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim * 2
        };

        let forward_config = lstm_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::lstm(input_dim, args.hidden_dim, forward_config, vb.clone())?;

        let backward_config = lstm_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::lstm(input_dim, args.hidden_dim, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    let mut input = vb.get((args.batch_size, args.seq_len, args.input_dim), "input")?;

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.combine_states_to_tensor(&forward_states, &backward_states)?;
    }
    Ok((input, answer))
}

fn test_lstm() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 1,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "lstm_test.pt",
    };

    let (output, answer) = run_lstm(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn test_nlayer_lstm() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 3,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "lstm_nlayer_test.pt",
    };

    let (output, answer) = run_lstm(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn test_bi_lstm() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 1,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "bi_lstm_test.pt",
    };

    let (output, answer) = run_bidirectional_lstm(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn test_nlayer_bi_lstm() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 3,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "bi_lstm_nlayer_test.pt",
    };

    let (output, answer) = run_bidirectional_lstm(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn run_gru(args: Args) -> Result<(Tensor, Tensor)> {
    let vb = args.load_pt()?;
    let answer = vb.get((args.batch_size, args.seq_len, args.hidden_dim), "output")?;

    let mut layers = Vec::with_capacity(args.layers);

    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim
        };
        let config = gru_config(layer_idx, rnn::Direction::Forward);
        let gru = candle_nn::gru(input_dim, args.hidden_dim, config, vb.clone())?;
        layers.push(gru);
    }

    let mut input = vb.get((args.batch_size, args.seq_len, args.input_dim), "input")?;

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    Ok((input, answer))
}

fn run_bidirectional_gru(args: Args) -> Result<(Tensor, Tensor)> {
    let vb = args.load_pt()?;
    let answer = vb.get(
        (args.batch_size, args.seq_len, args.hidden_dim * 2),
        "output",
    )?;

    let mut layers = Vec::with_capacity(args.layers);
    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim * 2
        };

        let forward_config = gru_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::gru(input_dim, args.hidden_dim, forward_config, vb.clone())?;

        let backward_config = gru_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::gru(input_dim, args.hidden_dim, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    let mut input = vb.get((args.batch_size, args.seq_len, args.input_dim), "input")?;

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.combine_states_to_tensor(&forward_states, &backward_states)?;
    }

    Ok((input, answer))
}

fn test_gru() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 1,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "gru_test.pt",
    };
    let (output, answer) = run_gru(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn test_bi_gru() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 1,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "bi_gru_test.pt",
    };

    let (output, answer) = run_bidirectional_gru(args)?;
    assert_tensor(&output, &answer, 3, 0.000001)
}

fn test_nlayer_gru() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 3,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "gru_nlayer_test.pt",
    };
    let (output, answer) = run_gru(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn test_nlayer_bi_gru() -> Result<()> {
    let args = Args {
        input_dim: IN_DIM,
        hidden_dim: HIDDEN_DIM,
        layers: 3,
        batch_size: BATCH_SIZE,
        seq_len: SEQ_LEN,
        pt: "bi_gru_nlayer_test.pt",
    };
    let (output, answer) = run_bidirectional_gru(args)?;
    assert_tensor(&output, &answer, 3, 0.0000001)
}

fn main() -> Result<()> {
    println!("test lstm");
    test_lstm()?;

    println!("test bi lstm");
    test_bi_lstm()?;

    println!("test gru");
    test_gru()?;

    println!("test_bi_gru");
    test_bi_gru()?;

    println!("test_nlayer_lstm");
    test_nlayer_lstm()?;

    println!("test_nlayer_bi_lstm");
    test_nlayer_bi_lstm()?;

    println!("test_nlayer_gru");
    test_nlayer_gru()?;

    println!("test_nlayer_bi_gru");
    test_nlayer_bi_gru()?;

    Ok(())
}

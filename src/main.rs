use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{rnn, GRUConfig, LSTMConfig, VarBuilder, RNN};
use std::path::Path;

static IN_DIM: usize = 10;
static HIDDEN_DIM: usize = 20;

fn load_pt<P: AsRef<Path>>(path: P) -> Result<VarBuilder<'static>> {
    Ok(VarBuilder::from_pth(path, DType::F32, &Device::Cpu)?)
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

fn test_lstm() {
    let vb = load_pt("lstm_test.pt").unwrap();
    let forward_config = LSTMConfig::default();
    let lstm = rnn::lstm(IN_DIM, HIDDEN_DIM, forward_config, vb.clone()).unwrap();

    let input = vb.get((5, 3, 10), "input").unwrap();
    let answer = vb.get((5, 3, 20), "output").unwrap();

    let states = lstm.seq(&input).unwrap();
    let output = lstm.states_to_tensor(&states).unwrap();
    assert_tensor(&output, &answer, 3, 0.0000001).unwrap();
}

fn test_bi_lstm() {
    let vb = load_pt("bi_lstm_test.pt").unwrap();
    let forward_config = LSTMConfig::default();
    let backward_config = {
        let mut config = LSTMConfig::default();
        config.direction = rnn::Direction::Backward;
        config
    };

    let foward_lstm = rnn::lstm(IN_DIM, HIDDEN_DIM, forward_config, vb.clone()).unwrap();
    let backward_lstm = rnn::lstm(IN_DIM, HIDDEN_DIM, backward_config, vb.clone()).unwrap();

    let input = vb.get((5, 3, 10), "input").unwrap();
    let answer = vb.get((5, 3, 40), "output").unwrap();

    let forward_states = foward_lstm.seq(&input).unwrap();
    let backward_states = backward_lstm.seq(&input).unwrap();
    let output = foward_lstm
        .combine_states_to_tensor(&forward_states, &backward_states)
        .unwrap();

    assert_tensor(&output, &answer, 3, 0.0000001).unwrap();
}

fn test_gru() {
    let vb = load_pt("gru_test.pt").unwrap();
    let forward_config = GRUConfig::default();
    let gru = rnn::gru(IN_DIM, HIDDEN_DIM, forward_config, vb.clone()).unwrap();

    let input = vb.get((5, 3, 10), "input").unwrap();
    let answer = vb.get((5, 3, 20), "output").unwrap();

    let states = gru.seq(&input).unwrap();
    let output = gru.states_to_tensor(&states).unwrap();
    assert_tensor(&output, &answer, 3, 0.0000001).unwrap();
}

fn test_bi_gru() {
    let vb = load_pt("bi_gru_test.pt").unwrap();
    let forward_config = GRUConfig::default();
    let backward_config = {
        let mut config = GRUConfig::default();
        config.direction = rnn::Direction::Backward;
        config
    };

    let foward_gru = rnn::gru(IN_DIM, HIDDEN_DIM, forward_config, vb.clone()).unwrap();
    let backward_gru = rnn::gru(IN_DIM, HIDDEN_DIM, backward_config, vb.clone()).unwrap();

    let input = vb.get((5, 3, 10), "input").unwrap();
    let answer = vb.get((5, 3, 40), "output").unwrap();

    let forward_states = foward_gru.seq(&input).unwrap();
    let backward_states = backward_gru.seq(&input).unwrap();
    let output = foward_gru
        .combine_states_to_tensor(&forward_states, &backward_states)
        .unwrap();

    assert_tensor(&output, &answer, 3, 0.0000001).unwrap();
}

fn main() {
    println!("test lstm");
    test_lstm();

    println!("test bi lstm");
    test_bi_lstm();

    println!("test gru");
    test_gru();

    println!("test_bi_gru");
    test_bi_gru();
}

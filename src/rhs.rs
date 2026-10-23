use crate::{
    bp::to_probabilities,
    util::{from_compl, to_compl},
};
use num_complex::Complex;

pub trait Rhs<const SZ_MSG: usize> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut Vec<[f64; SZ_MSG]>,
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: &Vec<i32>,
        sz_msg: usize,
        chk_sz: usize,
        connections: &Vec<usize>,
    );
    fn normalize(&mut self);
}

pub struct EqRhs<const SZ_MSG: usize> {
    pub value: i32,
}

impl<const SZ_MSG: usize> Rhs<SZ_MSG> for EqRhs<SZ_MSG> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut Vec<[f64; SZ_MSG]>,
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: &Vec<i32>,
        sz_msg: usize,
        chk_sz: usize,
        connections: &Vec<usize>,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[connections[i]];
            for v in 0..sz_msg as u32 {
                let v_m = from_compl(v, sz_msg) * coeff;
                let v_acc = to_compl(self.value - v_m, sz_chk);
                out_msgs[i][v as usize] += llo[i][v_acc].re;
            }
        }
    }
    fn normalize(&mut self) {}
}

// Represents low <= x <= high inequality correct with probability p
pub struct IneqRhs<const SZ_MSG: usize> {
    pub low: i32,
    pub high: i32,
    pub p: f64,
}

impl<const SZ_MSG: usize> Rhs<SZ_MSG> for IneqRhs<SZ_MSG> {
    // Bug fixed by gpt-5.6-sol.
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut Vec<[f64; SZ_MSG]>,
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: &Vec<i32>,
        sz_msg: usize,
        chk_sz: usize,
        connections: &Vec<usize>,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[connections[i]];
            let total: f64 = llo[i].iter().map(|value| value.re).sum();
            for v in 0..sz_msg as u32 {
                let v_m = from_compl(v, sz_msg) * coeff;
                let mut inside = 0.0;
                for value in self.low..=self.high {
                    let v_acc = to_compl(value - v_m, sz_chk);
                    inside += llo[i][v_acc].re;
                }
                let outside = (total - inside).max(0.0);
                out_msgs[i][v as usize] += self.p * inside + (1.0 - self.p) * outside;
            }
        }
    }
    fn normalize(&mut self) {}
}

pub struct ProbValueRhs<const SZ_MSG: usize> {
    pub values: Vec<(i32, f64)>,
    pub p: f64,
}

impl<const SZ_MSG: usize> Rhs<SZ_MSG> for ProbValueRhs<SZ_MSG> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut Vec<[f64; SZ_MSG]>,
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: &Vec<i32>,
        sz_msg: usize,
        chk_sz: usize,
        connections: &Vec<usize>,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[connections[i]];
            for v in 0..sz_msg as u32 {
                for (value, p) in &self.values {
                    let v_m = from_compl(v, sz_msg) * coeff;
                    let v_acc = to_compl(value - v_m, sz_chk);
                    out_msgs[i][v as usize] += p * llo[i][v_acc].re;
                }
            }
            to_probabilities(&mut out_msgs[i]);
            let q = (1.0 - self.p) / (sz_msg as f64);
            for v in 0..sz_msg {
                out_msgs[i][v] *= self.p;
                out_msgs[i][v] += q;
            }
        }
    }

    fn normalize(&mut self) {
        let sum: f64 = *self
            .values
            .iter()
            .map(|(_v, p)| p)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        self.values.iter_mut().for_each(|x| x.1 /= sum);
    }
}

pub struct ValueRhs<const SZ_MSG: usize> {
    pub values: Vec<(i32, f64)>,
}

impl<const SZ_MSG: usize> Rhs<SZ_MSG> for ValueRhs<SZ_MSG> {
    fn compute_msg_from_sumdist(
        &self,
        out_msgs: &mut Vec<[f64; SZ_MSG]>,
        llo: Vec<Vec<Complex<f64>>>,
        coeffs: &Vec<i32>,
        sz_msg: usize,
        chk_sz: usize,
        connections: &Vec<usize>,
    ) {
        let sz_chk = 1 << chk_sz;
        for i in 0..out_msgs.len() {
            let coeff = coeffs[connections[i]];
            for v in 0..sz_msg as u32 {
                for (value, p) in &self.values {
                    let v_m = from_compl(v, sz_msg) * coeff;
                    let v_acc = to_compl(value - v_m, sz_chk);
                    out_msgs[i][v as usize] += p * llo[i][v_acc].re;
                }
            }
        }
    }

    fn normalize(&mut self) {
        let sum: f64 = *self
            .values
            .iter()
            .map(|(_v, p)| p)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        self.values.iter_mut().for_each(|x| x.1 /= sum);
    }
}

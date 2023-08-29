use std::convert::TryInto;
use std::iter;

use halo2_base::{
    halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{AssignedCell, Cell, Chip, Layouter, Region, Value},
        plonk::{
            Advice, Any, Assigned, Column, ConstraintSystem, Error, Expression, Fixed, Selector,
        },
        poly::Rotation,
    },
    utils::value_to_option,
};

use super::{
    primitives::{Absorbing, Domain, Mds, Spec, Squeezing, State},
    PaddedWord, PermuteChip, PoseidonInstructions, PoseidonSpongeInstructions,
};

/// Trait for a variable in the circuit.
pub trait Var<'v, F: FieldExt>:
    Clone + std::fmt::Debug + From<PoseidonAssignedValue<'v, F>>
{
    /// The cell at which this variable was allocated.
    fn cell(&self) -> Cell;

    /// The value allocated to this variable.
    fn value(&self) -> Value<F>;
}

/// Configuration for a [`Pow5Chip`].
#[derive(Clone, Debug)]
pub struct Pow5Config<F: FieldExt, const WIDTH: usize, const RATE: usize> {
    pub(crate) state: [Column<Advice>; WIDTH],
    partial_sbox: Column<Advice>,
    rc_a: [Column<Fixed>; WIDTH],
    rc_b: [Column<Fixed>; WIDTH],
    s_full: Selector,
    s_partial: Selector,
    s_partial_single: Selector,
    s_pad_and_add: Selector,

    half_full_rounds: usize,
    half_partial_rounds: usize,
    alpha: [u64; 4],
    round_constants: Vec<[F; WIDTH]>,
    m_reg: Mds<F, WIDTH>,
    m_inv: Mds<F, WIDTH>,
}

/// A Poseidon chip using an $x^5$ S-Box.
///
/// The chip is implemented using a single round per row for full rounds, and two rounds
/// per row for partial rounds.
#[derive(Clone, Debug)]
pub struct Pow5Chip<F: FieldExt, const WIDTH: usize, const RATE: usize> {
    config: Pow5Config<F, WIDTH, RATE>,
}

impl<F: FieldExt, const WIDTH: usize, const RATE: usize> Pow5Chip<F, WIDTH, RATE> {
    /// Configures this chip for use in a circuit.
    ///
    /// # Side-effects
    ///
    /// All columns in `state` will be equality-enabled.
    //
    // TODO: Does the rate need to be hard-coded here, or only the width? It probably
    // needs to be known wherever we implement the hashing gadget, but it isn't strictly
    // necessary for the permutation.
    pub fn configure<S: Spec<F, WIDTH, RATE>>(
        meta: &mut ConstraintSystem<F>,
        state: [Column<Advice>; WIDTH],
        partial_sbox: Column<Advice>,
        rc_a: [Column<Fixed>; WIDTH],
        rc_b: [Column<Fixed>; WIDTH],
    ) -> Pow5Config<F, WIDTH, RATE> {
        assert_eq!(RATE, WIDTH - 1);
        // Generate constants for the Poseidon permutation.
        // This gadget requires R_F and R_P to be even.
        assert!(S::full_rounds() & 1 == 0);
        //assert!(S::partial_rounds() & 1 == 0);
        let half_full_rounds = S::full_rounds() / 2;
        let half_partial_rounds = S::partial_rounds() / 2;
        let (round_constants, m_reg, m_inv) = S::constants();

        // This allows state words to be initialized (by constraining them equal to fixed
        // values), and used in a permutation from an arbitrary region. rc_a is used in
        // every permutation round, while rc_b is empty in the initial and final full
        // rounds, so we use rc_b as "scratch space" for fixed values (enabling potential
        // layouter optimisations).
        for column in iter::empty()
            .chain(state.iter().cloned().map(Column::<Any>::from))
            .chain(rc_b.iter().cloned().map(Column::<Any>::from))
        {
            meta.enable_equality(column);
        }

        let s_full = meta.complex_selector();
        let s_partial = meta.complex_selector();
        let s_partial_single = meta.complex_selector();
        let s_pad_and_add = meta.complex_selector();

        let alpha = [5, 0, 0, 0];
        let pow_5 = |v: Expression<F>| {
            let v2 = v.clone() * v.clone();
            v2.clone() * v2 * v
        };

        meta.create_gate("full round", |meta| {
            let s_full = meta.query_selector(s_full);

            (0..WIDTH)
                .map(|next_idx| {
                    let state_next = meta.query_advice(state[next_idx], Rotation::next());
                    let expr = (0..WIDTH)
                        .map(|idx| {
                            let state_cur = meta.query_advice(state[idx], Rotation::cur());
                            let rc_a = meta.query_fixed(rc_a[idx], Rotation::cur());
                            pow_5(state_cur + rc_a) * m_reg[next_idx][idx]
                        })
                        .reduce(|acc, term| acc + term)
                        .expect("WIDTH > 0");
                    s_full.clone() * (expr - state_next)
                })
                .collect::<Vec<_>>()
        });

        meta.create_gate("partial rounds", |meta| {
            let cur_0 = meta.query_advice(state[0], Rotation::cur());
            let mid_0 = meta.query_advice(partial_sbox, Rotation::cur());

            let rc_a0 = meta.query_fixed(rc_a[0], Rotation::cur());
            let rc_b0 = meta.query_fixed(rc_b[0], Rotation::cur());

            let s_partial = meta.query_selector(s_partial);

            use halo2_base::halo2_proofs::plonk::VirtualCells;
            let mid = |idx: usize, meta: &mut VirtualCells<F>| {
                let mid = mid_0.clone() * m_reg[idx][0];
                (1..WIDTH).fold(mid, |acc, cur_idx| {
                    let cur = meta.query_advice(state[cur_idx], Rotation::cur());
                    let rc_a = meta.query_fixed(rc_a[cur_idx], Rotation::cur());
                    acc + (cur + rc_a) * m_reg[idx][cur_idx]
                })
            };

            let next = |idx: usize, meta: &mut VirtualCells<F>| {
                (0..WIDTH)
                    .map(|next_idx| {
                        let next = meta.query_advice(state[next_idx], Rotation::next());
                        next * m_inv[idx][next_idx]
                    })
                    .reduce(|acc, next| acc + next)
                    .expect("WIDTH > 0")
            };

            let partial_round_linear = |idx: usize, meta: &mut VirtualCells<F>| {
                let rc_b = meta.query_fixed(rc_b[idx], Rotation::cur());
                mid(idx, meta) + rc_b - next(idx, meta)
            };

            std::iter::empty()
                // state[0] round a
                .chain(Some(pow_5(cur_0 + rc_a0) - mid_0.clone()))
                // state[0] round b
                .chain(Some(pow_5(mid(0, meta) + rc_b0) - next(0, meta)))
                .chain((1..WIDTH).map(|idx| partial_round_linear(idx, meta)))
                .map(|exp| s_partial.clone() * exp)
                .collect::<Vec<_>>()
        });

        meta.create_gate("partial round single", |meta| {
            let s_partial_single = meta.query_selector(s_partial_single);

            (0..WIDTH)
                .map(|next_idx| {
                    let state_next = meta.query_advice(state[next_idx], Rotation::next());
                    let cur_0 = meta.query_advice(state[0], Rotation::cur());
                    let rc_a0 = meta.query_fixed(rc_a[0], Rotation::cur());

                    let expr = pow_5(cur_0 + rc_a0) * m_reg[next_idx][0];
                    let expr = expr
                        + (1..WIDTH)
                            .map(|idx| {
                                let state_cur = meta.query_advice(state[idx], Rotation::cur());
                                let rc_a = meta.query_fixed(rc_a[idx], Rotation::cur());
                                (state_cur + rc_a) * m_reg[next_idx][idx]
                            })
                            .reduce(|acc, term| acc + term)
                            .expect("WIDTH > 0");
                    s_partial_single.clone() * (expr - state_next)
                })
                .collect::<Vec<_>>()
        });

        meta.create_gate("pad-and-add", |meta| {
            let initial_state_rate = meta.query_advice(state[RATE], Rotation::prev());
            let output_state_rate = meta.query_advice(state[RATE], Rotation::next());

            let s_pad_and_add = meta.query_selector(s_pad_and_add);

            let pad_and_add = |idx: usize| {
                let initial_state = meta.query_advice(state[idx], Rotation::prev());
                let input = meta.query_advice(state[idx], Rotation::cur());
                let output_state = meta.query_advice(state[idx], Rotation::next());

                // We pad the input by storing the required padding in fixed columns and
                // then constraining the corresponding input columns to be equal to it.
                initial_state + input - output_state
            };

            (0..RATE)
                .map(pad_and_add)
                // The capacity element is never altered by the input.
                .chain(Some(initial_state_rate - output_state_rate))
                .map(|exp| s_pad_and_add.clone() * exp)
                .collect::<Vec<_>>()
        });

        Pow5Config {
            state,
            partial_sbox,
            rc_a,
            rc_b,
            s_full,
            s_partial,
            s_partial_single,
            s_pad_and_add,
            half_full_rounds,
            half_partial_rounds,
            alpha,
            round_constants,
            m_reg,
            m_inv,
        }
    }

    /// Construct a [`Pow5Chip`].
    pub fn construct(config: Pow5Config<F, WIDTH, RATE>) -> Self {
        Pow5Chip { config }
    }
}

impl<F: FieldExt, const WIDTH: usize, const RATE: usize> Chip<F> for Pow5Chip<F, WIDTH, RATE> {
    type Config = Pow5Config<F, WIDTH, RATE>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<'v, F: FieldExt, S: Spec<F, 3, 2>> PermuteChip<'v, F, S, 3, 2> for Pow5Chip<F, 3, 2> {
    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let state = [0; 3].map(|_| meta.advice_column());
        let partial_sbox = meta.advice_column();
        let constants = [0; 6].map(|_| meta.fixed_column());

        Pow5Chip::configure::<S>(
            meta,
            state,
            partial_sbox,
            constants[..3].try_into().unwrap(), //rc_a
            constants[3..].try_into().unwrap(), //rc_b
        )
    }

    fn construct(config: Self::Config) -> Self {
        Self::construct(config)
    }
}

impl<'v, F: FieldExt, S: Spec<F, WIDTH, RATE>, const WIDTH: usize, const RATE: usize>
    PoseidonInstructions<'v, F, S, WIDTH, RATE> for Pow5Chip<F, WIDTH, RATE>
{
    type Word = StateWord<'v, F>;

    fn permute(
        &self,
        layouter: &mut impl Layouter<F>,
        initial_state: &State<Self::Word, WIDTH>,
    ) -> Result<State<Self::Word, WIDTH>, Error> {
        let config = self.config();

        layouter.assign_region(
            || "permute state",
            |mut region| {
                // Load the initial state into this region.
                let state = Pow5State::load(&mut region, config, initial_state)?;

                let state = (0..config.half_full_rounds).fold(Ok(state), |res, r| {
                    res.and_then(|state| state.full_round(&mut region, config, r, r))
                })?;

                let state = (0..config.half_partial_rounds).fold(Ok(state), |res, r| {
                    res.and_then(|state| {
                        state.partial_round(
                            &mut region,
                            config,
                            config.half_full_rounds + 2 * r,
                            config.half_full_rounds + r,
                        )
                    })
                })?;

                let state = if config.half_partial_rounds * 2 < S::partial_rounds() {
                    state.partial_round_single(
                        &mut region,
                        config,
                        config.half_full_rounds + 2 * config.half_partial_rounds,
                        config.half_full_rounds + config.half_partial_rounds,
                    )
                } else {
                    Ok(state)
                }?;

                let odd_offset = S::partial_rounds() & 1;

                let state = (0..config.half_full_rounds).fold(Ok(state), |res, r| {
                    res.and_then(|state| {
                        state.full_round(
                            &mut region,
                            config,
                            config.half_full_rounds
                                + 2 * config.half_partial_rounds
                                + odd_offset
                                + r,
                            config.half_full_rounds + config.half_partial_rounds + odd_offset + r,
                        )
                    })
                })?;

                Ok(state.0)
            },
        )
    }
}

impl<
        'v,
        F: FieldExt,
        S: Spec<F, WIDTH, RATE>,
        D: Domain<F, RATE>,
        const WIDTH: usize,
        const RATE: usize,
    > PoseidonSpongeInstructions<'v, F, S, D, WIDTH, RATE> for Pow5Chip<F, WIDTH, RATE>
{
    fn initial_state(
        &self,
        layouter: &mut impl Layouter<F>,
    ) -> Result<State<Self::Word, WIDTH>, Error> {
        let config = self.config();
        let state = layouter.assign_region(
            || format!("initial state for domain {}", D::name()),
            |mut region| {
                let mut state = Vec::with_capacity(WIDTH);
                let mut load_state_word = |i: usize, value: F| -> Result<_, Error> {
                    let var = region.assign_advice(config.state[i], 0, Value::known(value));

                    state.push(StateWord(var));

                    Ok(())
                };

                for i in 0..RATE {
                    load_state_word(i, F::zero())?;
                }
                load_state_word(RATE, D::initial_capacity_element())?;

                Ok(state)
            },
        )?;

        // let state: State<Self::Word, WIDTH> = (0..WIDTH).map(|_| Self::Word::default()).collect().try_into().unwrap();
        // // Ok([Self::Word::default(); WIDTH])
        // Ok(state)
        Ok(state.try_into().unwrap())
    }

    fn add_input(
        &self,
        layouter: &mut impl Layouter<F>,
        initial_state: &State<Self::Word, WIDTH>,
        has_absorbed: bool,
        input: &Absorbing<PaddedWord<F>, RATE>,
    ) -> Result<State<Self::Word, WIDTH>, Error> {
        let config = self.config();
        layouter.assign_region(
            || format!("add input for domain {}", D::name()),
            |mut region| {
                config.s_pad_and_add.enable(&mut region, 1)?;

                // Load the initial state into this region.
                let initial_state = if has_absorbed {
                    let load_state_word = |i: usize| {
                        StateWord(
                            initial_state[i]
                                .0
                                .copy_advice(&mut region, config.state[i], 0),
                        )
                        .value()
                    };
                    let initial_state: Vec<Value<F>> = (0..WIDTH).map(load_state_word).collect();
                    initial_state
                } else {
                    let mut state = Vec::with_capacity(WIDTH);
                    let mut load_state_word = |i: usize, value: F| -> Result<_, Error> {
                        let var = region.assign_advice_from_constant(
                            || format!("state_{i}"),
                            config.state[i],
                            0,
                            value,
                        )?;
                        state.push(Value::known(value));
                        Ok(())
                    };

                    for i in 0..RATE {
                        load_state_word(i, F::zero())?;
                    }
                    load_state_word(RATE, D::initial_capacity_element())?;
                    state
                };

                // Load the input into this region.
                let load_input_word = |i: usize| {
                    let value = match input.0[i].clone() {
                        Some(PaddedWord::Message(word)) => {
                            word.copy_advice(&mut region, config.state[i], 1);
                            StateWord(word).value()
                        }
                        Some(PaddedWord::Padding(padding_value)) => {
                            // todo!("assign fixed as below");
                            let cell = region.assign_fixed(config.rc_b[i], 1, padding_value);
                            let val = Value::known(padding_value);
                            let assigned_cell = region.assign_advice(config.state[i], 1, val);
                            region.constrain_equal(&assigned_cell.cell(), &cell);
                            val
                        }
                        _ => panic!("Input is not padded"),
                    };
                    value
                };
                let input: Vec<Value<F>> = (0..RATE).map(load_input_word).collect();
                
                // let load_input_word = |i: usize| {
                //     let constraint_var: AssignedCell<&Assigned<F>, F> = match input.0[i].clone() {
                //         Some(PaddedWord::Message(word)) => word,
                //         Some(PaddedWord::Padding(padding_value)) => {
                //             // unimplemented!()
                //             region.assign_fixed(config.rc_b[i], 1, Value::known(padding_value))
                //         }
                //         _ => panic!("Input is not padded"),
                //     };
                //     constraint_var.copy_advice(&mut region, config.state[i], 1)
                // };
                // let input: Vec<_> = (0..RATE).map(load_input_word).map(StateWord).collect();
                // let input = input?;

                // Constrain the output.
                let constrain_output_word = |i: usize| {
                    region.assign_advice(config.state[i], 2, {
                        if let Some(inp) = input.get(i) {
                            initial_state[i] + inp
                        } else {
                            initial_state[i]
                        }
                    })
                };

                let output: Vec<_> = (0..WIDTH)
                    .map(constrain_output_word)
                    .map(StateWord)
                    .collect();
                Ok(output.try_into().unwrap())
            },
        )
    }

    fn get_output(state: &State<Self::Word, WIDTH>) -> Squeezing<Self::Word, RATE> {
        Squeezing(
            state[..RATE]
                .iter()
                .map(|word| Some(word.clone()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}

/// A word in the Poseidon state.
#[derive(Clone, Debug)]
pub struct StateWord<'v, F: FieldExt>(pub PoseidonAssignedValue<'v, F>);
pub type PoseidonAssignedValue<'v, F> = AssignedCell<&'v Assigned<F>, F>;

impl<'v, F: FieldExt> From<StateWord<'v, F>> for PoseidonAssignedValue<'v, F> {
    fn from(state_word: StateWord<'v, F>) -> PoseidonAssignedValue<'v, F> {
        state_word.0
    }
}

impl<'v, F: FieldExt> From<PoseidonAssignedValue<'v, F>> for StateWord<'v, F> {
    fn from(cell_value: PoseidonAssignedValue<'v, F>) -> StateWord<'v, F> {
        StateWord(cell_value)
    }
}

impl<'v, F: FieldExt> Var<'v, F> for StateWord<'v, F> {
    fn cell(&self) -> Cell {
        *self.0.cell()
    }

    fn value(&self) -> Value<F> {
        Value::known(extract_value(self.0.clone()))
    }
}

fn extract_value<'v, F: FieldExt>(assigned_value: PoseidonAssignedValue<'v, F>) -> F {
    let assigned = **value_to_option(assigned_value.value()).unwrap();
    match assigned {
        halo2_base::halo2_proofs::plonk::Assigned::Zero => F::zero(),
        halo2_base::halo2_proofs::plonk::Assigned::Trivial(f) => f,
        _ => panic!("value should be trival"),
    }
}

#[derive(Debug)]
struct Pow5State<'v, F: FieldExt, const WIDTH: usize>([StateWord<'v, F>; WIDTH]);

impl<'v, F: FieldExt, const WIDTH: usize> Pow5State<'v, F, WIDTH> {
    fn full_round<const RATE: usize>(
        self,
        region: &mut Region<F>,
        config: &Pow5Config<F, WIDTH, RATE>,
        round: usize,
        offset: usize,
    ) -> Result<Self, Error> {
        Self::round(region, config, round, offset, config.s_full, |_| {
            let q =
                self.0.iter().enumerate().map(|(idx, word)| {
                    word.value() + Value::known(config.round_constants[round][idx])
                });
            let r: Vec<Value<F>> = q
                .map(|q| {
                    q.map(|q| {
                        q.pow(&config.alpha)
                    })
                })
                .collect();
            let m = &config.m_reg;
            let state = m.iter().map(|m_i| {
                r.iter()
                    .enumerate()
                    .fold(Value::known(F::zero()), |acc, (j, r_j)| {
                        acc + Value::known(m_i[j]) * r_j
                    })
            });

            Ok((round + 1, state.collect::<Vec<_>>().try_into().unwrap()))
        })
    }

    fn partial_round_single<const RATE: usize>(
        self,
        region: &mut Region<F>,
        config: &Pow5Config<F, WIDTH, RATE>,
        round: usize,
        offset: usize,
    ) -> Result<Self, Error> {
        Self::round(
            region,
            config,
            round,
            offset,
            config.s_partial_single,
            |_| {
                let r: Vec<Value<F>> = self
                    .0
                    .iter()
                    .enumerate()
                    .map(|(idx, word)| {
                        word.value()
                            .map(|v| v + config.round_constants[round][idx])
                            .map(|v| {
                                if idx == 0 { v.pow(&config.alpha) } else { v }
                            })
                    })
                    .collect();

                let m = &config.m_reg;
                let state = m.iter().map(|m_i| {
                    r.iter()
                        .enumerate()
                        .fold(Value::known(F::zero()), |acc, (j, r_j)| {
                            acc + Value::known(m_i[j]) * r_j
                        })
                });

                Ok((round + 1, state.collect::<Vec<_>>().try_into().unwrap()))
            },
        )
    }

    fn partial_round<const RATE: usize>(
        self,
        region: &mut Region<F>,
        config: &Pow5Config<F, WIDTH, RATE>,
        round: usize,
        offset: usize,
    ) -> Result<Self, Error> {
        Self::round(region, config, round, offset, config.s_partial, |region| {
            let m = &config.m_reg;
            let p: Vec<_> = self.0.iter().map(|word| word.value()).collect();

            let r_0 = (p[0] + Value::known(config.round_constants[round][0])).map(|v| {
                v.pow(&config.alpha)
            });
            let r_i = p[1..]
                .iter()
                .enumerate()
                .map(|(i, p_i)| Value::known(config.round_constants[round][i + 1]) + p_i);
            let r: Vec<_> = Some(r_0).into_iter().chain(r_i).collect();

            region.assign_advice(config.partial_sbox, offset, r[0]);

            let p_mid: Vec<_> = m
                .iter()
                .map(|m_i| {
                    m_i.iter()
                        .zip(r.iter())
                        .fold(Value::known(F::zero()), |acc, (m_ij, r_j)| {
                            acc + Value::known(*m_ij) * r_j
                        })
                })
                .collect();

            // Load the second round constants.
            let mut load_round_constant = |i: usize| {
                region.assign_fixed(config.rc_b[i], offset, config.round_constants[round + 1][i])
            };
            for i in 0..WIDTH {
                load_round_constant(i);
            }

            let r_0 = (p_mid[0] + Value::known(config.round_constants[round + 1][0]))
                .map(|v| v.pow(&config.alpha));
            let r_i = p_mid[1..]
                .iter()
                .enumerate()
                .map(|(i, p_i)| Value::known(config.round_constants[round + 1][i + 1]) + p_i);
            let r_mid: Vec<_> = Some(r_0).into_iter().chain(r_i).collect();

            let state: Vec<_> = m
                .iter()
                .map(|m_i| {
                    m_i.iter()
                        .zip(r_mid.iter())
                        .fold(Value::known(F::zero()), |acc, (m_ij, r_j)| {
                            acc + Value::known(*m_ij) * r_j
                        })
                })
                .collect();

            Ok((round + 2, state.try_into().unwrap()))
        })
    }

    fn load<const RATE: usize>(
        region: &mut Region<F>,
        config: &Pow5Config<F, WIDTH, RATE>,
        initial_state: &State<StateWord<F>, WIDTH>,
    ) -> Result<Self, Error> {
        let load_state_word = |i: usize| {
            // let ref this = initial_state[i].0;
            let column = config.state[i];
            let assigned_cell =
                region.assign_advice(column, 0, initial_state[i].0.value().map(|v| *v));
            region.constrain_equal(&assigned_cell.cell(), &initial_state[i].0.cell());
            assigned_cell
        };

        let state: Vec<_> = (0..WIDTH).map(load_state_word).map(StateWord).collect();
        Ok(Pow5State(state.try_into().unwrap()))
    }

    fn round<const RATE: usize>(
        region: &mut Region<F>,
        config: &Pow5Config<F, WIDTH, RATE>,
        round: usize,
        offset: usize,
        round_gate: Selector,
        round_fn: impl FnOnce(&mut Region<F>) -> Result<(usize, [Value<F>; WIDTH]), Error>,
    ) -> Result<Self, Error> {
        // Enable the required gate.
        round_gate.enable(region, offset)?;

        // Load the round constants.
        let mut load_round_constant = |i: usize| {
            region.assign_fixed(config.rc_a[i], offset, config.round_constants[round][i])
        };
        for i in 0..WIDTH {
            load_round_constant(i);
        }

        // Compute the next round's state.
        let (next_round, next_state) = round_fn(region)?;

        let next_state_word = |i: usize| {
            let value = next_state[i];
            let var = region.assign_advice(config.state[i], offset + 1, value);
            Ok(StateWord(var))
        };

        let next_state: Result<Vec<_>, _> = (0..WIDTH).map(next_state_word).collect();
        next_state.map(|next_state| Pow5State(next_state.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use crate::poseidon::primitives::pasta::{test_vectors, Fp};
    use halo2_base::halo2_proofs::halo2curves::group::ff::{Field, PrimeField};
    use halo2_base::halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use rand::rngs::OsRng;

    use super::{PoseidonInstructions, Pow5Chip, Pow5Config, StateWord};
    use crate::poseidon::{
        primitives::{self as poseidon, ConstantLength, P128Pow5T3, Spec},
        Hash,
    };
    use std::convert::TryInto;
    use std::marker::PhantomData;

    type OrchardNullifier = P128Pow5T3<Fp>;

    struct HashCircuit<
        S: Spec<Fp, WIDTH, RATE>,
        const WIDTH: usize,
        const RATE: usize,
        const L: usize,
    > {
        message: Option<[Fp; L]>,
        // For the purpose of this test, witness the result.
        // TODO: Move this into an instance column.
        output: Option<Fp>,
        _spec: PhantomData<S>,
    }

    impl<S: Spec<Fp, WIDTH, RATE>, const WIDTH: usize, const RATE: usize, const L: usize>
        Circuit<Fp> for HashCircuit<S, WIDTH, RATE, L>
    {
        type Config = Pow5Config<Fp, WIDTH, RATE>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                message: None,
                output: None,
                _spec: PhantomData,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> Pow5Config<Fp, WIDTH, RATE> {
            let state = (0..WIDTH).map(|_| meta.advice_column()).collect::<Vec<_>>();
            let partial_sbox = meta.advice_column();

            let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
            let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

            meta.enable_constant(rc_b[0]);

            Pow5Chip::configure::<S>(
                meta,
                state.try_into().unwrap(),
                partial_sbox,
                rc_a.try_into().unwrap(),
                rc_b.try_into().unwrap(),
            )
        }

        fn synthesize(
            &self,
            config: Pow5Config<Fp, WIDTH, RATE>,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let chip = Pow5Chip::construct(config.clone());

            let message = layouter.assign_region(
                || "load message",
                |mut region| {
                    let message_word = |i: usize| {
                        let value = self.message.map(|message_vals| message_vals[i]);
                        region.assign_advice(
                            config.state[i % WIDTH],
                            i / WIDTH,
                            if let Some(v) = value {
                                Value::known(v)
                            } else {
                                Value::unknown()
                            },
                        )
                    };

                    let message: Vec<_> = (0..L).map(message_word).collect();
                    Ok(message.try_into().unwrap())
                },
            )?;

            let hasher = Hash::<_, _, S, ConstantLength<L>, WIDTH, RATE>::init(
                chip,
                layouter.namespace(|| "init"),
            )?;
            let output = hasher.hash(layouter.namespace(|| "hash"), message)?;

            layouter.assign_region(
                || "constrain output",
                |mut region| {
                    let expected_var = region.assign_advice(
                        config.state[0],
                        0,
                        if let Some(v) = self.output {
                            Value::known(v)
                        } else {
                            Value::unknown()
                        },
                    );
                    region.constrain_equal(output.cell(), expected_var.cell());
                    Ok(())
                },
            )
        }
    }

    #[test]
    fn poseidon_hash() {
        let rng = OsRng;

        let message = [Fp::random(rng), Fp::random(rng)];
        let output =
            poseidon::Hash::<_, OrchardNullifier, ConstantLength<2>, 3, 2>::init().hash(message);

        let k = 6;
        let circuit = HashCircuit::<OrchardNullifier, 3, 2, 2> {
            message: Some(message),
            output: Some(output),
            _spec: PhantomData,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }

    #[test]
    fn poseidon_hash_longer_input() {
        let rng = OsRng;

        let message = [
            Fp::random(rng),
            Fp::random(rng),
            Fp::random(rng),
            Fp::random(rng),
        ];
        let output =
            poseidon::Hash::<_, OrchardNullifier, ConstantLength<4>, 3, 2>::init().hash(message);

        let k = 7;
        let circuit = HashCircuit::<OrchardNullifier, 3, 2, 4> {
            message: Some(message),
            output: Some(output),
            _spec: PhantomData,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }

    #[test]
    fn hash_test_vectors() {
        for tv in test_vectors::fp::hash() {
            let message = [
                Fp::from_repr(tv.input[0]).unwrap(),
                Fp::from_repr(tv.input[1]).unwrap(),
            ];
            let output = poseidon::Hash::<_, OrchardNullifier, ConstantLength<2>, 3, 2>::init()
                .hash(message);

            let k = 6;
            let circuit = HashCircuit::<OrchardNullifier, 3, 2, 2> {
                message: Some(message),
                output: Some(output),
                _spec: PhantomData,
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            assert_eq!(prover.verify(), Ok(()));
        }
    }

    #[cfg(feature = "print_layout")]
    #[test]
    fn print_poseidon_chip() {
        use plotters::prelude::*;

        let root =
            BitMapBackend::new("layouts/poseidon-chip-layout.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("Poseidon Chip Layout", ("sans-serif", 60))
            .unwrap();

        let circuit = HashCircuit::<OrchardNullifier, 3, 2, 2> {
            message: None,
            output: None,
            _spec: PhantomData,
        };
        halo2_proofs::dev::CircuitLayout::default()
            .render(6, &circuit, &root)
            .unwrap();
    }
}

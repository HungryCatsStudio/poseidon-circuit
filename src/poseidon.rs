//! The Poseidon algebraic hash function.

use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;

use halo2_base::halo2_proofs::{
    arithmetic::{Field, FieldExt},
    circuit::{AssignedCell, Chip, Layouter},
    plonk::{ConstraintSystem, Error},
};

mod pow5;
pub use pow5::{Pow5Chip, Pow5Config, StateWord, Var};

pub mod primitives;
use primitives::{Absorbing, ConstantLength, Domain, Spec, SpongeMode, Squeezing, State};
use std::fmt::Debug as DebugT;

use self::pow5::PoseidonAssignedValue;

/// A word from the padded input to a Poseidon sponge.
#[derive(Clone, Debug)]
pub enum PaddedWord<'v, F: Field> {
    /// A message word provided by the prover.
    Message(PoseidonAssignedValue<'v, F>),
    /// A padding word, that will be fixed in the circuit parameters.
    Padding(F),
}

/// This trait is the interface to chips that implement a permutation.
pub trait PermuteChip<'v, F: FieldExt, S: Spec<F, T, RATE>, const T: usize, const RATE: usize>:
    Chip<F> + Clone + DebugT + PoseidonInstructions<'v, F, S, T, RATE>
{
    /// Configure the permutation chip.
    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config;

    /// Get a chip from its config.
    fn construct(config: Self::Config) -> Self;
}

/// The set of circuit instructions required to use the Poseidon permutation.
pub trait PoseidonInstructions<
    'v,
    F: FieldExt,
    S: Spec<F, T, RATE>,
    const T: usize,
    const RATE: usize,
>: Chip<F>
{
    /// Variable representing the word over which the Poseidon permutation operates.
    type Word: Clone
        + fmt::Debug
        + From<PoseidonAssignedValue<'v, F>>
        + Into<PoseidonAssignedValue<'v, F>>
        + Send
        + Sync;

    /// Applies the Poseidon permutation to the given state.
    fn permute(
        &self,
        layouter: &mut impl Layouter<F>,
        initial_states: &State<Self::Word, T>,
    ) -> Result<State<Self::Word, T>, Error>;
}

/// The set of circuit instructions required to use the [`Sponge`] and [`Hash`] gadgets.
///
/// [`Hash`]: self::Hash
pub trait PoseidonSpongeInstructions<
    'v,
    F: FieldExt,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
>: PoseidonInstructions<'v, F, S, T, RATE>
{
    /// Returns the initial empty state for the given domain.
    fn initial_state(&self, layouter: &mut impl Layouter<F>)
        -> Result<State<Self::Word, T>, Error>;

    /// Adds the given input to the state.
    fn add_input(
        &self,
        layouter: &mut impl Layouter<F>,
        initial_state: &State<Self::Word, T>,
        input: &Absorbing<PaddedWord<F>, RATE>,
    ) -> Result<State<Self::Word, T>, Error>;

    /// Extracts sponge output from the given state.
    fn get_output(state: &State<Self::Word, T>) -> Squeezing<Self::Word, RATE>;
}

/// A word over which the Poseidon permutation operates.
#[derive(Debug)]
pub struct Word<
    'v,
    F: FieldExt,
    PoseidonChip: PoseidonInstructions<'v, F, S, T, RATE>,
    S: Spec<F, T, RATE>,
    const T: usize,
    const RATE: usize,
> {
    inner: PoseidonChip::Word,
}

impl<
        'v,
        F: FieldExt,
        PoseidonChip: PoseidonInstructions<'v, F, S, T, RATE>,
        S: Spec<F, T, RATE>,
        const T: usize,
        const RATE: usize,
    > Word<'v, F, PoseidonChip, S, T, RATE>
{
    /// The word contained in this gadget.
    pub fn inner(&self) -> PoseidonChip::Word {
        self.inner.clone()
    }

    /// Construct a [`Word`] gadget from the inner word.
    pub fn from_inner(inner: PoseidonChip::Word) -> Self {
        Self { inner }
    }
}

fn poseidon_sponge<
    'v,
    F: FieldExt,
    PoseidonChip: PoseidonSpongeInstructions<'v, F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
>(
    chip: &PoseidonChip,
    mut layouter: impl Layouter<F>,
    state: &mut State<PoseidonChip::Word, T>,
    input: &Absorbing<PaddedWord<F>, RATE>,
) -> Result<Squeezing<PoseidonChip::Word, RATE>, Error> {
    *state = chip.add_input(&mut layouter, state, input)?;
    *state = chip.permute(&mut layouter, state)?;
    Ok(PoseidonChip::get_output(state))
}

/// A Poseidon sponge.
#[derive(Debug)]
pub struct Sponge<
    'v,
    F: FieldExt,
    PoseidonChip: PoseidonSpongeInstructions<'v, F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    M: SpongeMode,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
> {
    chip: PoseidonChip,
    mode: M,
    state: State<PoseidonChip::Word, T>,
    _marker: PhantomData<D>,
}

impl<
        'v,
        F: FieldExt,
        PoseidonChip: PoseidonSpongeInstructions<'v, F, S, D, T, RATE>,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Sponge<'v, F, PoseidonChip, S, Absorbing<PaddedWord<'v, F>, RATE>, D, T, RATE>
{
    /// Absorbs an element into the sponge.
    pub fn absorb(
        &mut self,
        mut layouter: impl Layouter<F>,
        value: PaddedWord<F>,
    ) -> Result<(), Error> {
        for entry in self.mode.0.iter_mut() {
            if entry.is_none() {
                *entry = Some(value);
                return Ok(());
            }
        }

        // We've already absorbed as many elements as we can
        let _ = poseidon_sponge(
            &self.chip,
            layouter.namespace(|| "PoseidonSponge"),
            &mut self.state,
            &self.mode,
        )?;
        self.mode = Absorbing::init_with(value);

        Ok(())
    }

    /// Transitions the sponge into its squeezing state.
    #[allow(clippy::type_complexity)]
    pub fn finish_absorbing(
        mut self,
        mut layouter: impl Layouter<F>,
    ) -> Result<
        Sponge<'v, F, PoseidonChip, S, Squeezing<PoseidonChip::Word, RATE>, D, T, RATE>,
        Error,
    > {
        let mode = poseidon_sponge(
            &self.chip,
            layouter.namespace(|| "PoseidonSponge"),
            &mut self.state,
            &self.mode,
        )?;

        Ok(Sponge {
            chip: self.chip,
            mode,
            state: self.state,
            _marker: PhantomData::default(),
        })
    }
}

impl<
        'v,
        F: FieldExt,
        PoseidonChip: PoseidonSpongeInstructions<'v, F, S, D, T, RATE>,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Sponge<'v, F, PoseidonChip, S, Squeezing<PoseidonChip::Word, RATE>, D, T, RATE>
{
    /// Squeezes an element from the sponge.
    pub fn squeeze(
        &mut self,
        mut layouter: impl Layouter<F>,
    ) -> Result<PoseidonAssignedValue<'v, F>, Error> {
        loop {
            for entry in self.mode.0.iter_mut() {
                if let Some(inner) = entry.take() {
                    return Ok(inner.into());
                }
            }
        }
    }
}

/// A Poseidon hash function, built around a sponge.
#[derive(Debug)]
pub struct Hash<
    'v,
    F: FieldExt,
    PoseidonChip: PoseidonSpongeInstructions<'v, F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
    const L: usize,
> {
    sponge: Sponge<'v, F, PoseidonChip, S, Absorbing<PaddedWord<'v, F>, RATE>, D, T, RATE>,
}

impl<
        'v,
        F: FieldExt,
        PoseidonChip: PoseidonSpongeInstructions<'v, F, S, D, T, RATE>,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
        const L: usize,
    > Hash<'v, F, PoseidonChip, S, D, T, RATE, L>
{
    /// Initializes a new hasher.
    pub fn hash(
        chip: PoseidonChip,
        message: [PoseidonAssignedValue<'v, F>; L],
        mut layouter: impl Layouter<F>,
    ) -> Result<PoseidonAssignedValue<'v, F>, Error> {
        let mut layouter = layouter;
        let initial_state = chip.initial_state(&mut layouter)?;

        let mut sponge = Sponge {
            chip,
            mode: Absorbing(
                (0..RATE)
                    .map(|_| None)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            ),
            state: initial_state,
            _marker: PhantomData::default(),
        };

        for (i, value) in message
            .into_iter()
            .map(PaddedWord::Message)
            .chain(<ConstantLength<L> as Domain<F, RATE>>::padding(L).map(PaddedWord::Padding))
            .enumerate()
        {
            let mut layouter = layouter.namespace(|| format!("absorb_{i}"));
            for entry in sponge.mode.0.iter_mut() {
                if entry.is_none() {
                    *entry = Some(value);
                    continue;
                }
            }

            // We've already absorbed as many elements as we can
            let chip: &PoseidonChip = &sponge.chip;
            let mut layouter = layouter.namespace(|| "PoseidonSponge");
            let state: &mut State<PoseidonChip::Word, T> = &mut sponge.state;
            let input: &Absorbing<PaddedWord<F>, RATE> = &sponge.mode;
            *state = chip.add_input(&mut layouter, state, input)?;
            *state = chip.permute(&mut layouter, state)?;

            sponge.mode = Absorbing::init_with(value);
        }

        // finish absorbing
        let mut layouter = layouter.namespace(|| "finish absorbing");
        let mode = {
            let chip: &PoseidonChip = &sponge.chip;
            let mut layouter = layouter.namespace(|| "PoseidonSponge");
            let state: &mut State<PoseidonChip::Word, T> = &mut sponge.state;
            let input: &Absorbing<PaddedWord<F>, RATE> = &sponge.mode;
            *state = chip.add_input(&mut layouter, state, input)?;
            *state = chip.permute(&mut layouter, state)?;
            PoseidonChip::get_output(state)
        };

        let mut sponge = Sponge {
            chip: sponge.chip,
            mode,
            state: sponge.state,
            _marker: PhantomData::default(),
        };

        // finally squeeze
        sponge.squeeze(layouter.namespace(|| "squeeze"))
    }
}

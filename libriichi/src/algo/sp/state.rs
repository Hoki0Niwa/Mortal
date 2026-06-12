use super::CALC_SHANTEN_FN;
use super::tile::{DiscardTile, DrawTile, RequiredTile};
use crate::algo::shanten::DeltaCalculator;
use crate::tile::Tile;
use crate::{must_tile, t, tu8};

use ahash::AHashMap;
use tinyvec::ArrayVec;

/// Shanten numbers of one-tile modifications of a hand, memoized by the hand
/// itself. The board state plays no role here, so entries can be shared
/// between all `State`s that only differ in `tiles_in_wall`.
pub(super) type ShantenDeltaCache = AHashMap<[u8; 34], [i8; 34]>;

/// Delta-based equivalent of `CALC_SHANTEN_FN` for hands that differ from the
/// `DeltaCalculator`'s base hand by one tile.
#[cfg(feature = "sp_reproduce_cpp_ver")]
fn delta_shanten(delta: &DeltaCalculator, tid: usize, plus: bool) -> i8 {
    delta.calc_normal_delta(tid, plus)
}
#[cfg(not(feature = "sp_reproduce_cpp_ver"))]
fn delta_shanten(delta: &DeltaCalculator, tid: usize, plus: bool) -> i8 {
    delta.calc_all_delta(tid, plus)
}

/// `CALC_SHANTEN_FN` of `tehai` with one tile of `tid` added, for all 34 tids.
fn shanten_deltas_plus(tehai: &[u8; 34], tehai_len_div3: u8) -> [i8; 34] {
    let delta = DeltaCalculator::new(tehai, tehai_len_div3);
    std::array::from_fn(|tid| delta_shanten(&delta, tid, true))
}

/// `CALC_SHANTEN_FN` of `tehai` with one tile of `tid` removed, for all 34
/// tids present in the hand. Other entries are unusable and never read.
fn shanten_deltas_minus(tehai: &[u8; 34], tehai_len_div3: u8) -> [i8; 34] {
    let delta = DeltaCalculator::new(tehai, tehai_len_div3);
    std::array::from_fn(|tid| {
        if tehai[tid] > 0 {
            delta_shanten(&delta, tid, false)
        } else {
            i8::MAX
        }
    })
}

/// Mutable state of both the hand and the board.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct State {
    // hand
    pub(super) tehai: [u8; 34],
    pub(super) akas_in_hand: [bool; 3],

    // global
    pub(super) tiles_in_wall: [u8; 34],
    pub(super) akas_in_wall: [bool; 3],
    pub(super) n_extra_tsumo: u8,
}

/// Mutable state of both the hand and the board.
#[derive(Clone)]
pub struct InitState {
    // hand
    pub tehai: [u8; 34],
    pub akas_in_hand: [bool; 3],

    // global
    pub tiles_seen: [u8; 34],
    pub akas_seen: [bool; 3],
}

impl From<InitState> for State {
    fn from(
        InitState {
            tehai,
            akas_in_hand,
            tiles_seen,
            akas_seen,
        }: InitState,
    ) -> Self {
        let mut tiles_in_wall = tiles_seen;
        let mut akas_in_wall = akas_seen;
        tiles_in_wall.iter_mut().for_each(|v| *v = 4 - *v);
        akas_in_wall.iter_mut().for_each(|v| *v = !*v);
        Self {
            tehai,
            akas_in_hand,
            tiles_in_wall,
            akas_in_wall,
            n_extra_tsumo: 0,
        }
    }
}

/// Packs the whole state into a single buffer so the hasher gets one slice
/// write instead of many per-field writes. The packing is injective, so this
/// is consistent with the derived `PartialEq`.
impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut buf = [0_u8; 70];
        buf[..34].copy_from_slice(&self.tehai);
        buf[34..68].copy_from_slice(&self.tiles_in_wall);
        buf[68] = self.akas_in_hand[0] as u8
            | (self.akas_in_hand[1] as u8) << 1
            | (self.akas_in_hand[2] as u8) << 2
            | (self.akas_in_wall[0] as u8) << 3
            | (self.akas_in_wall[1] as u8) << 4
            | (self.akas_in_wall[2] as u8) << 5;
        buf[69] = self.n_extra_tsumo;
        state.write(&buf);
    }
}

impl State {
    pub(super) const fn discard(&mut self, tile: Tile) {
        self.tehai[tile.deaka().as_usize()] -= 1;
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_hand[0] = false,
            tu8!(5pr) => self.akas_in_hand[1] = false,
            tu8!(5sr) => self.akas_in_hand[2] = false,
            _ => (),
        }
    }

    pub(super) const fn undo_discard(&mut self, tile: Tile) {
        self.tehai[tile.deaka().as_usize()] += 1;
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_hand[0] = true,
            tu8!(5pr) => self.akas_in_hand[1] = true,
            tu8!(5sr) => self.akas_in_hand[2] = true,
            _ => (),
        }
    }

    pub(super) const fn deal(&mut self, tile: Tile) {
        self.tiles_in_wall[tile.deaka().as_usize()] -= 1;
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_wall[0] = false,
            tu8!(5pr) => self.akas_in_wall[1] = false,
            tu8!(5sr) => self.akas_in_wall[2] = false,
            _ => (),
        }
        self.undo_discard(tile);
    }

    pub(super) const fn undo_deal(&mut self, tile: Tile) {
        self.discard(tile);
        self.tiles_in_wall[tile.deaka().as_usize()] += 1;
        match tile.as_u8() {
            tu8!(5mr) => self.akas_in_wall[0] = true,
            tu8!(5pr) => self.akas_in_wall[1] = true,
            tu8!(5sr) => self.akas_in_wall[2] = true,
            _ => (),
        }
    }

    pub(super) fn get_discard_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
        cache: &mut ShantenDeltaCache,
    ) -> ArrayVec<[DiscardTile; 14]> {
        let mut discard_tiles = ArrayVec::default();

        let deltas = cache
            .entry(self.tehai)
            .or_insert_with(|| shanten_deltas_minus(&self.tehai, tehai_len_div3));
        for (tid, &count) in self.tehai.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let shanten_after = deltas[tid];
            let shanten_diff = shanten_after - shanten;

            let tile = match tid as u8 {
                tu8!(5m) if self.akas_in_hand[0] && count == 1 => t!(5mr),
                tu8!(5p) if self.akas_in_hand[1] && count == 1 => t!(5pr),
                tu8!(5s) if self.akas_in_hand[2] && count == 1 => t!(5sr),
                _ => must_tile!(tid),
            };

            discard_tiles.push(DiscardTile { tile, shanten_diff });
        }

        discard_tiles
    }

    pub(super) fn get_draw_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
        cache: &mut ShantenDeltaCache,
    ) -> ArrayVec<[DrawTile; 37]> {
        let mut draw_tiles = ArrayVec::default();

        let deltas = cache
            .entry(self.tehai)
            .or_insert_with(|| shanten_deltas_plus(&self.tehai, tehai_len_div3));
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let shanten_after = deltas[tid];
            let shanten_diff = shanten_after - shanten;

            let tile = must_tile!(tid);
            match (tid as u8, self.akas_in_wall) {
                (tu8!(5m), [true, _, _]) | (tu8!(5p), [_, true, _]) | (tu8!(5s), [_, _, true]) => {
                    if count >= 2 {
                        draw_tiles.push(DrawTile {
                            tile,
                            count: count - 1,
                            shanten_diff,
                        });
                    }
                    draw_tiles.push(DrawTile {
                        tile: tile.akaize(),
                        count: 1,
                        shanten_diff,
                    });
                }
                _ => draw_tiles.push(DrawTile {
                    tile,
                    count,
                    shanten_diff,
                }),
            }
        }

        draw_tiles
    }

    pub(super) fn get_required_tiles(
        &self,
        tehai_len_div3: u8,
        cache: &mut ShantenDeltaCache,
    ) -> ArrayVec<[RequiredTile; 34]> {
        let shanten = CALC_SHANTEN_FN(&self.tehai, tehai_len_div3);
        let mut required_tiles = ArrayVec::default();

        let deltas = cache
            .entry(self.tehai)
            .or_insert_with(|| shanten_deltas_plus(&self.tehai, tehai_len_div3));
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let shanten_after = deltas[tid];
            if shanten_after < shanten {
                required_tiles.push(RequiredTile {
                    tile: must_tile!(tid),
                    count,
                });
            }
        }

        required_tiles
    }

    pub(super) fn sum_left_tiles(&self) -> u8 {
        self.tiles_in_wall.iter().sum()
    }
}

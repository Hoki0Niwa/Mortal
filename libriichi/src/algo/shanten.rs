//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//!
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>

use crate::tuz;
use std::io::prelude::*;
use std::sync::LazyLock;

use flate2::read::GzDecoder;

const JIHAI_TABLE_SIZE: usize = 78_032;
const SUHAI_TABLE_SIZE: usize = 1_940_777;

static JIHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_jihai.bin.gz"),
        JIHAI_TABLE_SIZE,
    )
});
static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_suhai.bin.gz"),
        SUHAI_TABLE_SIZE,
    )
});

fn read_table(gzipped: &[u8], length: usize) -> Vec<[u8; 10]> {
    let mut gz = GzDecoder::new(gzipped);
    let mut raw = vec![];
    gz.read_to_end(&mut raw).unwrap();

    let mut ret = Vec::with_capacity(length);
    let mut entry = [0; 10];
    for (i, b) in raw.into_iter().enumerate() {
        entry[i * 2 % 10] = b & 0b1111;
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), length);

    ret
}

pub fn ensure_init() {
    assert_eq!(JIHAI_TABLE.len(), JIHAI_TABLE_SIZE);
    assert_eq!(SUHAI_TABLE.len(), SUHAI_TABLE_SIZE);
}

/// Min-plus convolution of two partial distance vectors, truncated to at most
/// `m` sets and one pair. This is exactly the merge step `add_suhai` has
/// always performed; it is associative and commutative, so partial merges can
/// be combined in any order without changing the result.
fn merge(lhs: &mut [u8; 10], tab: &[u8; 10], m: usize) {
    for j in (5..=(5 + m)).rev() {
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
        }
        lhs[j] = sht;
    }

    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            sht = sht.min(lhs[k] + tab[j - k]);
        }
        lhs[j] = sht;
    }
}

/// Like `merge`, but only computes the final element `m + 5`, which is the
/// only one `calc_normal` reads after the last merge.
fn merge_final(lhs: &[u8; 10], tab: &[u8; 10], m: usize) -> u8 {
    let j = m + 5;
    let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
    for k in 5..j {
        sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
    }
    sht
}

fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();
    merge(lhs, &tab, m);
}

fn add_jihai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = JIHAI_TABLE.get(index).copied().unwrap_or_default();
    lhs[m + 5] = merge_final(lhs, &tab, m);
}

fn sum_tiles(tiles: &[u8]) -> usize {
    tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)
}

/// `len_div3` must be within [0, 4].
#[must_use]
pub fn calc_normal(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let len_div3 = len_div3 as usize;

    let mut ret = SUHAI_TABLE
        .get(sum_tiles(&tiles[..9]))
        .copied()
        .unwrap_or_default();
    add_suhai(&mut ret, sum_tiles(&tiles[9..2 * 9]), len_div3);
    add_suhai(&mut ret, sum_tiles(&tiles[2 * 9..3 * 9]), len_div3);
    add_jihai(&mut ret, sum_tiles(&tiles[3 * 9..]), len_div3);

    (ret[5 + len_div3] as i8) - 1
}

#[must_use]
pub fn calc_chitoi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;
    tiles.iter().filter(|&&c| c > 0).for_each(|&c| {
        kinds += 1;
        if c >= 2 {
            pairs += 1;
        }
    });

    let redunct = 7_u8.saturating_sub(kinds) as i8;
    7 - pairs + redunct - 1
}

#[must_use]
pub fn calc_kokushi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;

    tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C]
        .iter()
        .map(|&i| tiles[i])
        .filter(|&c| c > 0)
        .for_each(|c| {
            kinds += 1;
            if c >= 2 {
                pairs += 1;
            }
        });

    let redunct = (pairs > 0) as i8;
    14 - kinds - redunct - 1
}

/// Answers shanten queries for hands that differ from a fixed base hand by
/// exactly one tile, reusing partial merge results across queries.
///
/// For each tile group (m/p/s/z), the merge of the table entries of the other
/// three groups is precomputed once. A query then only needs one table lookup
/// for the modified group plus one final-element merge, instead of the four
/// lookups and three full merges `calc_normal` performs. Since the merge is an
/// associative and commutative integer min-plus convolution, the results are
/// guaranteed to be identical to calling `calc_normal` / `calc_all` on the
/// modified hand.
pub struct DeltaCalculator {
    len_div3: usize,
    counts: [u8; 34],
    indices: [usize; 4],
    /// `loo[g]` = merged table entries of all groups except `g`.
    loo: [[u8; 10]; 4],
    chitoi_kinds: u8,
    chitoi_pairs: u8,
    kokushi_kinds: u8,
    kokushi_pairs: u8,
}

const YAOCHUU: [usize; 13] = tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C];

impl DeltaCalculator {
    #[must_use]
    pub fn new(tiles: &[u8; 34], len_div3: u8) -> Self {
        let m = len_div3 as usize;
        let indices = [
            sum_tiles(&tiles[..9]),
            sum_tiles(&tiles[9..2 * 9]),
            sum_tiles(&tiles[2 * 9..3 * 9]),
            sum_tiles(&tiles[3 * 9..]),
        ];
        let entries = [
            SUHAI_TABLE.get(indices[0]).copied().unwrap_or_default(),
            SUHAI_TABLE.get(indices[1]).copied().unwrap_or_default(),
            SUHAI_TABLE.get(indices[2]).copied().unwrap_or_default(),
            JIHAI_TABLE.get(indices[3]).copied().unwrap_or_default(),
        ];

        let mut p01 = entries[0];
        merge(&mut p01, &entries[1], m);
        let mut p012 = p01;
        merge(&mut p012, &entries[2], m);
        let mut s23 = entries[2];
        merge(&mut s23, &entries[3], m);
        let mut loo0 = entries[1];
        merge(&mut loo0, &s23, m);
        let mut loo1 = entries[0];
        merge(&mut loo1, &s23, m);
        let mut loo2 = p01;
        merge(&mut loo2, &entries[3], m);

        let mut chitoi_kinds = 0;
        let mut chitoi_pairs = 0;
        for &c in tiles {
            if c > 0 {
                chitoi_kinds += 1;
                if c >= 2 {
                    chitoi_pairs += 1;
                }
            }
        }
        let mut kokushi_kinds = 0;
        let mut kokushi_pairs = 0;
        for &c in YAOCHUU.map(|i| tiles[i]).iter() {
            if c > 0 {
                kokushi_kinds += 1;
                if c >= 2 {
                    kokushi_pairs += 1;
                }
            }
        }

        Self {
            len_div3: m,
            counts: *tiles,
            indices,
            loo: [loo0, loo1, loo2, p012],
            chitoi_kinds,
            chitoi_pairs,
            kokushi_kinds,
            kokushi_pairs,
        }
    }

    /// (group, base-5 digit weight of `tid` within its group)
    const fn group_of(tid: usize) -> (usize, usize) {
        if tid < 27 {
            (tid / 9, 5_usize.pow(8 - (tid % 9) as u32))
        } else {
            (3, 5_usize.pow(6 - (tid - 27) as u32))
        }
    }

    /// `calc_normal` of the base hand with one `tid` added (`plus = true`) or
    /// removed (`plus = false`).
    #[must_use]
    pub fn calc_normal_delta(&self, tid: usize, plus: bool) -> i8 {
        let (g, w) = Self::group_of(tid);
        let index = if plus {
            self.indices[g] + w
        } else {
            self.indices[g] - w
        };
        let tab = if g == 3 {
            JIHAI_TABLE.get(index).copied().unwrap_or_default()
        } else {
            SUHAI_TABLE.get(index).copied().unwrap_or_default()
        };
        (merge_final(&self.loo[g], &tab, self.len_div3) as i8) - 1
    }

    fn calc_chitoi_delta(&self, tid: usize, plus: bool) -> i8 {
        let c = self.counts[tid];
        let (kinds, pairs) = if plus {
            (
                self.chitoi_kinds + (c == 0) as u8,
                self.chitoi_pairs + (c == 1) as u8,
            )
        } else {
            (
                self.chitoi_kinds - (c == 1) as u8,
                self.chitoi_pairs - (c == 2) as u8,
            )
        };
        let redunct = 7_u8.saturating_sub(kinds) as i8;
        7 - pairs as i8 + redunct - 1
    }

    fn calc_kokushi_delta(&self, tid: usize, plus: bool) -> i8 {
        let is_yaochuu = matches!(tid, 0 | 8 | 9 | 17 | 18 | 26..=33);
        let (kinds, pairs) = if !is_yaochuu {
            (self.kokushi_kinds, self.kokushi_pairs)
        } else {
            let c = self.counts[tid];
            if plus {
                (
                    self.kokushi_kinds + (c == 0) as u8,
                    self.kokushi_pairs + (c == 1) as u8,
                )
            } else {
                (
                    self.kokushi_kinds - (c == 1) as u8,
                    self.kokushi_pairs - (c == 2) as u8,
                )
            }
        };
        let redunct = (pairs > 0) as i8;
        14 - kinds as i8 - redunct - 1
    }

    /// `calc_all` of the base hand with one `tid` added (`plus = true`) or
    /// removed (`plus = false`).
    #[must_use]
    pub fn calc_all_delta(&self, tid: usize, plus: bool) -> i8 {
        let mut shanten = self.calc_normal_delta(tid, plus);
        if shanten <= 0 || self.len_div3 < 4 {
            return shanten;
        }

        shanten = shanten.min(self.calc_chitoi_delta(tid, plus));
        if shanten > 0 {
            shanten.min(self.calc_kokushi_delta(tid, plus))
        } else {
            shanten
        }
    }
}

#[must_use]
pub fn calc_all(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let mut shanten = calc_normal(tiles, len_div3);
    if shanten <= 0 || len_div3 < 4 {
        return shanten;
    }

    shanten = shanten.min(calc_chitoi(tiles));
    if shanten > 0 {
        shanten.min(calc_kokushi(tiles))
    } else {
        shanten
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;
    use rand::prelude::*;

    #[test]
    fn delta_equivalence() {
        let mut rng = StdRng::seed_from_u64(0xd05a);
        let mut pool: Vec<u8> = (0..34).flat_map(|t| [t; 4]).collect();

        for i in 0..30_000 {
            let len_div3 = (i % 4 + 1) as u8;
            let n_tiles = 3 * len_div3 as usize + 1 + (i / 4) % 2;
            let (hand_pool, _) = pool.partial_shuffle(&mut rng, n_tiles);
            let mut tiles = [0_u8; 34];
            for &t in hand_pool.iter() {
                tiles[t as usize] += 1;
            }

            let delta = DeltaCalculator::new(&tiles, len_div3);
            for tid in 0..34 {
                if tiles[tid] < 4 {
                    tiles[tid] += 1;
                    assert_eq!(
                        delta.calc_normal_delta(tid, true),
                        calc_normal(&tiles, len_div3),
                        "normal +{tid} {tiles:?} {len_div3}"
                    );
                    assert_eq!(
                        delta.calc_all_delta(tid, true),
                        calc_all(&tiles, len_div3),
                        "all +{tid} {tiles:?} {len_div3}"
                    );
                    tiles[tid] -= 1;
                }
                if tiles[tid] > 0 {
                    tiles[tid] -= 1;
                    assert_eq!(
                        delta.calc_normal_delta(tid, false),
                        calc_normal(&tiles, len_div3),
                        "normal -{tid} {tiles:?} {len_div3}"
                    );
                    assert_eq!(
                        delta.calc_all_delta(tid, false),
                        calc_all(&tiles, len_div3),
                        "all -{tid} {tiles:?} {len_div3}"
                    );
                    tiles[tid] += 1;
                }
            }
        }
    }

    #[test]
    fn calc_3n_plus_1() {
        let tehai = hand("1111m 333p 222s 444z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        let tehai = hand("147m 258p 369s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 6);
        let tehai = hand("468m 33346p 7s").unwrap();
        assert_eq!(calc_all(&tehai, 3), 2);
        let tehai = hand("147m 258p 3s").unwrap();
        assert_eq!(calc_all(&tehai, 2), 4);
        let tehai = hand("4455s").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        let tehai = hand("7z").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        let tehai = hand("15559m 19p 19s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        let tehai = hand("9999m 6677p 88s 355z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("19m 19p 159s 123456z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
    }

    #[test]
    fn calc_3n_plus_2() {
        let tehai = hand("2344456m 14p 127s 2z 7p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        let tehai = hand("2344456m 14p 127s 2z 5p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("344455667p 1139s 9m").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("344455667p 1139s 9p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        let tehai = hand("122334m 678p 37s 22z 5s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        let tehai = hand("122334m 678p 12s 22z 4s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        let tehai = hand("12223456m 78889p 2m").unwrap();
        assert_eq!(calc_all(&tehai, 4), -1);
        let tehai = hand("34778p").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        let tehai = hand("34s").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        let tehai = hand("55m").unwrap();
        assert_eq!(calc_all(&tehai, 0), -1);
    }
}

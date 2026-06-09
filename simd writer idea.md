

To encode 8 'bit-wise' bytes and add `0x00` after any `0xff`s, we first load 8 bytes into a simd register with 0x00 between each element.

We get the following:

| 0  | 1    | 2  | 3    | 4  | 5    | 6  | 7    | 8  | 9    | a  | b    | c  | d    | e  | f    |
|----|------|----|------|----|------|----|------|----|------|----|------|----|------|----|------|
| b0 | 0x00 | b1 | 0x00 | b2 | 0x00 | b3 | 0x00 | b4 | 0x00 | b5 | 0x00 | b6 | 0x00 | b7 | 0x00 |

Because they all start as `u8`s we can actually get this by doing a simd cast to `u16`. Preferred to use the swizzle macro though to avoid endianness dependency.

Then we create a mask from an `eq` with `splat(0xff)`. In this example we assume `b1` and `b4` are `0xff`:

| 0  | 1    | 2  | 3    | 4  | 5    | 6  | 7    | 8  | 9    | a  | b    | c  | d    | e  | f    |
|----|------|----|------|----|------|----|------|----|------|----|------|----|------|----|------|
| b0 | 0x00 | b1 | 0x00 | b2 | 0x00 | b3 | 0x00 | b4 | 0x00 | b5 | 0x00 | b6 | 0x00 | b7 | 0x00 |
| 0  | 0    | 1  | 0    | 0  | 0    | 0  | 0    | 1  | 0    | 0  | 0    | 0  | 0    | 0  | 0    |

At this point we can clone the mask and `to_bitmask` -> `count_ones()` + 8 to get the length of the final sequence

Then we rotate the original mask right once, moving the 1's in the mask from under the value to under the following `0x00`:

| 0  | 1    | 2  | 3    | 4  | 5    | 6  | 7    | 8  | 9    | a  | b    | c  | d    | e  | f    |
|----|------|----|------|----|------|----|------|----|------|----|------|----|------|----|------|
| b0 | 0x00 | b1 | 0x00 | b2 | 0x00 | b3 | 0x00 | b4 | 0x00 | b5 | 0x00 | b6 | 0x00 | b7 | 0x00 |
| 0  | 0    | 0  | 1    | 0  | 0    | 0  | 0    | 0  | 1    | 0  | 0    | 0  | 0    | 0  | 0    |

Then we 'or' that mask with an alternating pattern, making the mask a 1 at every index corresponding to a real value:

| 0  | 1    | 2  | 3    | 4  | 5    | 6  | 7    | 8  | 9    | a  | b    | c  | d    | e  | f    |
|----|------|----|------|----|------|----|------|----|------|----|------|----|------|----|------|
| b0 | 0x00 | b1 | 0x00 | b2 | 0x00 | b3 | 0x00 | b4 | 0x00 | b5 | 0x00 | b6 | 0x00 | b7 | 0x00 |
| 1  | 0    | 1  | 1    | 1  | 0    | 1  | 0    | 1  | 1    | 1  | 0    | 1  | 0    | 1  | 0    |

Now calling `store_select`
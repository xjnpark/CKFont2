import numpy as np
import random
import itertools
from jamo import h2j, j2hcj


filename_r = './labels/2000-common-hangul.txt'
filename_w = './labels/2000-common-hangul-split.txt'

with open(filename_w, 'wt',encoding='utf-8') as fw:

    with open(filename_r, 'rt', encoding='utf-8') as fr:
        
        s = fr.read()
        c = j2hcj(h2j(s))
        
        for i in range(len(c)):
            #s_c = c[i]
            fw.writelines('\n'.join(c[i]))#print(s)
            #print(s_c)
print("Done creates file : ", filename_w)


# syllables = np.array([chr(code) for code in range(44032, 55204)])
# syllables = syllables.reshape(19, 21, 28)
# #print(f"'ㄱ'과 관련된 마지막 음절 list : {syllables[0][0][0]}")

# def cj_generate():

#     cnt_1 = 0
#     cj =[]
#     filename_cj = 'cj_399.txt'

#     with open(filename_cj, 'w') as f:
#         for i in range(19) :
#             for j in range(21) :
#                 cj.append(syllables[i][j][0])
#                 #print(f"{syllables[i][j][0]}")
#                 f.writelines(cj)
#                 cnt_1 +=1
#     f.close()
#     print("Done", cnt_1)
#     return cj

# def cjo_generate():
    
#     cnt_2 = 0
#     cjo =[]
#     filename_cjo = 'cjo_11k.txt'
    
#     with open(filename_cjo, 'w') as f:
#         for i in range(19) :
#             for j in range(21) :
#                 for k in range(28) :
#                     cjo.append(syllables[i][j][k])
#                     f.writelines(cjo)
#                     cnt_2 +=1
#     # print(cnt_2)

#     f.close() 
#     print("Done", cnt_2)
#     return cjo

# # c = cj_generate()   
# # c = cjo_generate()   
# # s = random.sample(c, 10)
# # s=cjo[1]
# # print(s)


# # filename = './labels/cj_399.txt'
# # # filename = './labels/cjo_11k_10.txt'

# # print("Done selectes all and creates file : ", filename)

# # with open(filename, 'w', encoding="UTF8") as f:
# #     f.writelines('\n'.join(c))
# # f.close()
# # 초성(19)
# # 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
# # 중성(21)
# # 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
# # 중성(28)
# # 없음, 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
#  # 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'

# __all__ = ["split_syllable_char", "split_syllables",
#            "join_jamos", "join_jamos_char",
#            "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

# INITIAL = 0x001
# MEDIAL = 0x010
# FINAL = 0x100
# # CHAR_LISTS = { # 1st and 3rd comp are different
# #     INITIAL: list(map(chr, [
# #         0x1100, 0x1101, 0x1102, 0x1103, 0x1104, 0x1105,
# #         0x1106, 0x1107, 0x1108, 0x1109, 0x110A, 0x110B,
# #         0x110C, 0x110D, 0x110E, 0x110F, 0x1110, 0x1111,
# #         0x1112       
# #     ])),
# #     MEDIAL: list(map(chr, [
# #         0x1161, 0x1162, 0x1163, 0x1164, 0x1165, 0x1166,
# #         0x1167, 0x1168, 0x1169, 0x116A, 0x116B, 0x116C,
# #         0x116D, 0x116E, 0x116F, 0x1170, 0x1171, 0x1172,
# #         0x1173, 0x1174, 0x1175
# #     ])),
# #     FINAL: list(map(chr, [
# #         0x11A8, 0x11A9, 0x11AA, 0x11AB, 0x11AC, 0x11AD,
# #         0x11AE, 0xD7CF, 0x11AF, 0x11B0, 0x11B1, 0x11B2,
# #         0x11B3, 0x11B4, 0x11B5, 0x11B6, 0x11B7, 0x11B8,
# #         0x11B9, 0x11BA, 0x11BB, 0x11BC, 0x11BD, 0x11BE,
# #         0x11BF, 0x11C0, 0x11C1, 0x11C2, 
# #     ]))
# # }
# CHAR_LISTS = {# 1st and 3rd comp are same
#     INITIAL: list(map(chr, [
#         0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
#         0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
#         0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
#         0x314e
#     ])),
#     MEDIAL: list(map(chr, [
#         0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
#         0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
#         0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
#         0x3161, 0x3162, 0x3163
#     ])),
#     FINAL: list(map(chr, [
#         0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
#         0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
#         0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
#         0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
#         0x314c, 0x314d, 0x314e
#     ]))
# }
# CHAR_INITIALS = CHAR_LISTS[INITIAL]
# CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
# CHAR_FINALS = CHAR_LISTS[FINAL]
# CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
# CHARSET = set(itertools.chain(*CHAR_SETS.values()))
# CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
#                 for k, v in CHAR_LISTS.items()}

# #         0x1100, 0x1101, 0x1102, 0x1103, 0x1104, 0x1105,
# #         0x1106, 0x1107, 0x1108, 0x1109, 0x110a, 0x110b,
# #         0x110c, 0x110d, 0x110e, 0x110f, 0x1110, 0x1111,
# #         0x1112
#     # MEDIAL: list(map(chr, [
#     #     0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
#     #     0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
#     #     0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
#     #     0x3161, 0x3162, 0x3163
# def is_hangul_syllable(c):
#     return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


# def is_hangul_jamo(c):
#     return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


# def is_hangul_compat_jamo(c):
#     return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


# def is_hangul_jamo_exta(c):
#     return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


# def is_hangul_jamo_extb(c):
#     return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


# def is_hangul(c):
#     return (is_hangul_syllable(c) or
#             is_hangul_jamo(c) or
#             is_hangul_compat_jamo(c) or
#             is_hangul_jamo_exta(c) or
#             is_hangul_jamo_extb(c))


# def is_supported_hangul(c):
#     return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


# def check_hangul(c, jamo_only=False):
#     if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
#         raise ValueError(f"'{c}' is not a supported hangul character. "
#                          f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
#                          f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
#                          f"supported at the moment.")


# def get_jamo_type(c):
#     check_hangul(c)
#     assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
#     return sum(t for t, s in CHAR_SETS.items() if c in s)


# def split_syllable_char(c):
#     """
#     """
#     check_hangul(c)
#     if len(c) != 1:
#         raise ValueError("Input string must have exactly one character.")

#     init, med, final = None, None, None
#     if is_hangul_syllable(c):
#         offset = ord(c) - 0xac00
#         x = (offset - offset % 28) // 28
#         init, med, final = x // 21, x % 21, offset % 28
#         if not final:
#             final = None
#         else:
#             final -= 1
#     else:
#         pos = get_jamo_type(c)
#         if pos & INITIAL == INITIAL:
#             pos = INITIAL
#         elif pos & MEDIAL == MEDIAL:
#             pos = MEDIAL
#         elif pos & FINAL == FINAL:
#             pos = FINAL
#         idx = CHAR_INDICES[pos][c]
#         if pos == INITIAL:
#             init = idx
#         elif pos == MEDIAL:
#             med = idx
#         else:
#             final = idx
#     return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
#                  for pos, idx in
#                  zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


# def split_syllables(s, ignore_err=True, pad=None):
#     """
#     """

#     def try_split(c):
#         try:
#             return split_syllable_char(c)
#         except ValueError:
#             if ignore_err:
#                 return (c,)
#             raise ValueError(f"encountered an unsupported character: "
#                              f"{c} (0x{ord(c):x})")

#     s = map(try_split, s)
#     if pad is not None:
#         tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
#     else:
#         tuples = map(lambda x: filter(None, x), s)
#     return "".join(itertools.chain(*tuples))


# def join_jamos_char(init, med, final=None):
#     """
#     Combines jamos into a single syllable.

#     """
#     chars = (init, med, final)
#     for c in filter(None, chars):
#         check_hangul(c, jamo_only=True)

#     idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
#                 for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
#     init_idx, med_idx, final_idx = idx
#     # final index must be shifted once as
#     # final index with 0 points to syllables without final
#     final_idx = 0 if final_idx is None else final_idx + 1
#     return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


# def join_jamos(s, ignore_err=True):
#     """
#     """
#     last_t = 0
#     queue = []
#     new_string = ""

#     def flush(n=0):
#         new_queue = []
#         while len(queue) > n:
#             new_queue.append(queue.pop())
#         if len(new_queue) == 1:
#             if not ignore_err:
#                 raise ValueError(f"invalid jamo character: {new_queue[0]}")
#             result = new_queue[0]
#         elif len(new_queue) >= 2:
#             try:
#                 result = join_jamos_char(*new_queue)
#             except (ValueError, KeyError):
#                 # Invalid jamo combination
#                 if not ignore_err:
#                     raise ValueError(f"invalid jamo characters: {new_queue}")
#                 result = "".join(new_queue)
#         else:
#             result = None
#         return result

#     for c in s:
#         if c not in CHARSET:
#             if queue:
#                 new_c = flush() + c
#             else:
#                 new_c = c
#             last_t = 0
#         else:
#             t = get_jamo_type(c)
#             new_c = None
#             if t & FINAL == FINAL:
#                 if not (last_t == MEDIAL):
#                     new_c = flush()
#             elif t == INITIAL:
#                 new_c = flush()
#             elif t == MEDIAL:
#                 if last_t & INITIAL == INITIAL:
#                     new_c = flush(1)
#                 else:
#                     new_c = flush()
#             last_t = t
#             queue.insert(0, c)
#         if new_c:
#             new_string += new_c
#     if queue:
#         new_string += flush()
#     return new_string

# # s_split=split_syllables(c)

# # filename_split = './labels/cj_399_split.txt'
# # # filename_split = './labels/cjo_11k_10_split.txt'

# # with open(filename_split, 'w', encoding="UTF8") as f:
# #     f.writelines('\n'.join(s_split))
# # f.close()
# # print("Done split allselected and creates file : ", filename_split)

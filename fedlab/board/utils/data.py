def encode_int_array(arr: list[int]):
    arr = sorted(arr)
    bits = []
    for idx in arr:
        while idx > len(bits):
            bits.append(0)
        bits.append(1)
    while len(bits) % 4 != 0:
        bits.append(0)
    grouped_list = [bits[i:i + 4] for i in range(0, len(bits), 4)]
    hex_list = [hex(int(''.join(map(str, group)), 2))[2:] for group in grouped_list]
    hex_string = ''.join(hex_list)
    return hex_string

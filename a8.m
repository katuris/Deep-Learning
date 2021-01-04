alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
int = 0:25
char_int = containers.Map(alphabet,int)
seq_length = 1
dataX = []
dataY = []
for i = drange(0:24)
    seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
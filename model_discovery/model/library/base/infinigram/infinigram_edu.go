package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path"
	"strings"

	"infinigram/tokenizers"
)

// Wrapper around infini-gram model data.
type ModelData struct {
	suffixArray SuffixArray
	bytesData   TokenArray
	vocabSize   int
}

// Wrapper around infini-gram model predictions results.
type Prediction struct {
	distribution      []float32 // next-token probability distribution
	effectiveN        int       // length of the longest suffix used
	numRetrieved      int       // number of continuations retrieved
	numExtend         int       // in the retrievedSuffixes, number of additional tokens added
	retrievedSuffixes [][]int   // raw retrieved suffixes
}

// Will return the prediction of the next token distribution corresponding to the
// longest suffix in queryIds. For a suffix to be considered valid, there must be
// at least minMatches occurrences of it in the data. The retrieved suffixes will
// include numExtend extra tokens (set to 1 to just get the next token).
func (m *ModelData) NextTokenDistribution(queryIds []uint32, numExtend int, minMatches int) *Prediction {
	vocabSize := m.vocabSize
	suffixArray := m.suffixArray
	dataBytes := m.bytesData

	var bestQueryEnc []byte

	if len(queryIds) == 0 {
		bestQueryEnc = make([]byte, 0)
	} else {
		// perform binary search to find longest suffix
		left := 0
		right := len(queryIds) + 1

		for left < right {
			// the current candidate for the longest suffix length
			mid := (left + right) / 2
			queryIdsSuffix := queryIds[len(queryIds)-mid:]

			querySuffixEnc := intToByte(queryIdsSuffix)

			// check if, at this length, we get any matches
			numMatches := suffixArray.retrieveNum(dataBytes, querySuffixEnc)

			if numMatches >= minMatches {
				left = mid + 1
			} else {
				right = mid
			}
		}

		if left == 0 {
			// TODO: i don't think this should happen
			fmt.Println("none found")
			return &Prediction{nil, -1, 0, numExtend, make([][]int, 0)}
		}
		best_n := left - 1

		bestQueryEnc = intToByte(queryIds[len(queryIds)-best_n:])
	}

	substrings := suffixArray.retrieveSubstrings(dataBytes, bestQueryEnc, int64(numExtend))

	rawSuffixes := make([][]int, len(substrings))
	distr := make([]float32, vocabSize)
	total := 0
	for i, s := range substrings {
		retrievedSuffix := byteToInt(s)

		newIds := retrievedSuffix[len(retrievedSuffix)-numExtend:]
		newIds = append([]int{}, newIds...)

		// add to raw retrievals
		rawSuffixes[i] = newIds

		// populate distribution
		distr[newIds[0]] += 1
		total += 1
	}

	for i := range distr {
		distr[i] /= float32(total)
	}

	return &Prediction{distr, len(bestQueryEnc) / 2, total, numExtend, rawSuffixes}
}

// Will generate a sequence of numNewTokens tokens greedily using the longest matched
// suffix. For a suffix to be considered valid, there must be at least minMatches
// occurrences of it in the data. queryIds are the initial prompt tokens.
func (m *ModelData) GenerateGreedy(queryIds []uint32, numNewTokens, minMatches int) []uint32 {
	result := make([]uint32, 0, len(queryIds)+numNewTokens)
	result = append(result, queryIds...)

	for i := 0; i < numNewTokens; i++ {
		prediction := m.NextTokenDistribution(result, 1, minMatches)

		if prediction.numRetrieved == 0 {
			return result
		}

		newToken := uint32(argmax(prediction.distribution))
		result = append(result, newToken)
	}

	return result
}

// Same as GenerateGreedy, but will send intermediate results to the generatedTokens
func (m *ModelData) GenerateGreedyStream(queryIds []uint32, numNewTokens, minMatches int, generatedTokens chan<- []uint32) {
	defer close(generatedTokens)

	result := make([]uint32, 0, len(queryIds)+numNewTokens)
	result = append(result, queryIds...)

	for i := 0; i < numNewTokens; i++ {
		prediction := m.NextTokenDistribution(result, 1, minMatches)

		if prediction.numRetrieved == 0 {
			return
		}

		newToken := uint32(argmax(prediction.distribution))
		result = append(result, newToken)

		generatedTokens <- result
	}
}

// Creates the tokenized corpus and suffix array, saves them to outpath, and returns
// the model. If either the tokenized corpus or suffix array already exist, they will
// be loaded from disk. Set sentinalVal and sentinalSize to split documents.
// Tokenization is done in parallel using nWorkers. Set lineSplit to the token that
// separates documents in the input file (filename). Set tokenizerConfig to the path
// of the tokenizer configuration file. vocabSize is the size of the vocabulary.
// Creates a suffix array for each chunk of documents of size chunkSize.
func InitializeModel(filename, lineSplit, outpath, tokenizerConfig string, sentinalVal, sentinalSize, nWorkers, vocabSize, chunkSize int) (*ModelData, error) {
	// check whether tokenized data already exists
	dataPath := path.Join(outpath, "data.bin")
	_, err := os.Stat(dataPath)
	if err != nil {
		// tokenize data: streams documents from text file into binary file
		fmt.Println("Tokenizing data to disk")
		_, err := tokenizeMultiprocess(filename, lineSplit, outpath, tokenizerConfig, sentinalVal, sentinalSize, nWorkers)
		if err != nil {
			return nil, err
		}
	} else {
		fmt.Println("Tokenized data already found")
	}

	dataBytes, err := loadMMappedArray(dataPath)
	if err != nil {
		return nil, err
	}

	// check whether suffix array already exists
	saChunkPathsPath := path.Join(outpath, "suffix_array_paths.txt")

	_, err = os.Stat(saChunkPathsPath)
	if err == nil {
		fmt.Println("Suffix array(s) already found")

		saChunkPathsStr, err := readStringFromFile(saChunkPathsPath)
		if err != nil {
			return nil, err
		}

		suffixArray, err := makeMultiSuffixArray(strings.Split(saChunkPathsStr, "\n"))
		if err != nil {
			return nil, err
		}

		return &ModelData{
			suffixArray: suffixArray,
			bytesData:   dataBytes,
			vocabSize:   vocabSize,
		}, nil
	}

	fmt.Println("Creating suffix array(s)")
	offset := int64(0)
	currChunk := 0
	chunkBuffer := make([]byte, chunkSize)
	saCallback := func(chunkLength int) error {
		fmt.Printf("making chunk %d of size %d\n", currChunk, chunkLength)

		readValues := chunkBuffer[:chunkLength]

		unalignedSa := createUnalignedSuffixArray(readValues)

		saChunkPath := path.Join(outpath, fmt.Sprintf("suffix_array_%d.bin", currChunk))
		err = writeIndicesToFile(saChunkPath, unalignedSa, offset)
		if err != nil {
			return err
		}

		currChunk += 1
		offset += int64(chunkLength)

		return nil
	}
	err = documentIter(dataPath, sentinalSize, sentinalVal, chunkBuffer, saCallback)
	if err != nil {
		return nil, err
	}

	// get list of filenames
	numChunks := currChunk
	saChunkPaths := make([]string, numChunks)
	for i := 0; i < numChunks; i++ {
		saChunkPaths[i] = path.Join(outpath, fmt.Sprintf("suffix_array_%d.bin", i))
	}

	// save list of filenames to disk
	err = writeStringToFile(saChunkPathsPath, strings.Join(saChunkPaths, "\n"))
	if err != nil {
		return nil, err
	}

	suffixArray, err := makeMultiSuffixArray(saChunkPaths)
	if err != nil {
		return nil, err
	}

	return &ModelData{
		suffixArray: suffixArray,
		bytesData:   dataBytes,
		vocabSize:   vocabSize,
	}, nil
}

// Given a sequence of tokens (queryIds) will print the top-k most likely continuations using
// the longest possible suffix. The suffix must have at least minMatches occurrences in the data.
// modelData and tk are the model and tokenizer, respectively.
func InteractiveNextToken(queryIds []uint32, modelData *ModelData, tk *tokenizers.Tokenizer, top_k, minMatches int) {
	prediction := modelData.NextTokenDistribution(queryIds, 1, minMatches)

	if prediction.numRetrieved == 0 {
		fmt.Println("No continuations found")
		return
	}

	topIndices := intToUint32(argsort(prediction.distribution, true))
	if len(topIndices) > top_k {
		topIndices = topIndices[:top_k]
	}

	fullGeneration := append([]uint32{}, queryIds...)
	fullGeneration = append(fullGeneration, 0)
	for i, tkn_idx := range topIndices {
		prob := prediction.distribution[tkn_idx]

		fullGeneration[len(fullGeneration)-1] = tkn_idx

		total := prediction.numRetrieved
		fmt.Printf(
			"n=%d, p=%.3f (%d/%d), k=%d: %s\n",
			prediction.effectiveN,
			prob,
			int(prob*float32(total)),
			total,
			i,
			tk.Decode(fullGeneration, true),
		)
	}
}

// Given a sequence of tokens (queryIds) will greedily generate numNewTokens tokens using the
// longest possible suffix. The suffix must have at least minMatches occurrences in the data.
// modelData and tk are the model and tokenizer, respectively.
func InteractiveGenerateGreedy(queryIds []uint32, modelData *ModelData, tk *tokenizers.Tokenizer, numNewTokens, minMatches int) {
	generated_tokens := make(chan []uint32, 8)

	go modelData.GenerateGreedyStream(queryIds, numNewTokens, minMatches, generated_tokens)

	for tkns := range generated_tokens {
		fmt.Printf("====\n%s\n", tk.Decode(tkns, true))
	}
}

func main() {
	var _ = fmt.Printf

	var (
		filename        string
		outpath         string
		nWorkers        int
		tokenizerConfig string
		sentinalVal     int
		sentinalSize    int
		minMatches      int
		topK            int
		interactiveMode int
		numGenerate     int
		lineSplit       string
		maxMem          int
	)

	flag.StringVar(&filename, "train_file", "", "Path to training data")
	flag.StringVar(&lineSplit, "line_split", "\n", "String to split documents in training data file")
	flag.StringVar(&outpath, "out_dir", "", "Directory to save trained model")
	flag.IntVar(&nWorkers, "n_workers", 4, "Number of workers to use")
	flag.StringVar(&tokenizerConfig, "tokenizer_config", "tokenizer_gpt2.json", "Path to .json file containing tokenizer configuration")
	flag.IntVar(&sentinalVal, "sentinal_val", 0, "Value to add at the end of every document")
	flag.IntVar(&sentinalSize, "sentinal_size", 2, "Number of sentinals to add at the end of every document")
	flag.IntVar(&minMatches, "min_matches", 1, "Minimum number of continuations needed for suffix to be valid")

	flag.IntVar(&maxMem, "max_mem", 1024, "Maximum size (in MiB) of documents for each chunk")

	flag.IntVar(&interactiveMode, "interactive_mode", 0, "0: print the top-k best next-token continuations 1: greedily generate k tokens")
	flag.IntVar(&topK, "top_k", 8, "Number of most frequent continuations to print during interactive mode 0")
	flag.IntVar(&numGenerate, "num_generate", 32, "Number of new tokens to generate")

	flag.Parse()

	// load tokenizer
	tk, err := tokenizers.FromFile(tokenizerConfig)
	if err != nil {
		panic(err)
	}

	defer tk.Close()

	modelDataP, err := InitializeModel(
		filename,
		lineSplit,
		outpath,
		tokenizerConfig,
		sentinalVal,
		sentinalSize,
		nWorkers,
		int(tk.VocabSize()),
		maxMem*1024*1024,
	)
	if err != nil {
		panic(err)
	}

	modelData := *modelDataP

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("enter query: ")

		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("Error reading input:", err)
			continue
		}

		input = strings.TrimSuffix(input, "\n")
		input = strings.TrimSuffix(input, "\r")

		en, _ := tk.Encode(input, false)

		fmt.Println("encoded tokens:", en)

		if interactiveMode == 0 {
			InteractiveNextToken(en, &modelData, tk, topK, minMatches)
		} else if interactiveMode == 1 {
			InteractiveGenerateGreedy(en, &modelData, tk, numGenerate, minMatches)
		}
	}
}
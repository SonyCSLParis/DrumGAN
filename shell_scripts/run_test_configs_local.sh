realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

declare -a basic_configs=(
	"config_files/waveform/waveform_test_config.json"
	"config_files/stft/stft_test_config.json"
	"config_files/mag-if/mag-if_test_config.json"
	"config_files/mel/mel_test_config.json"
	"config_files/mfcc/mfcc_test_config.json"
	"config_files/cqt/cqt_test_config.json"
	"config_files/cq-nsgt/cq-nsgt_test_config.json"

)
# source "/Users/javier/.virtualenvs/main/bin/activate"
conda activate main


for i in ${basic_configs[@]};
	do
		echo "TESTING CONFIG FILE: $i"
		echo ""
		python train.py $1 -c `realpath $i` -s 10
done

echo "TESTS FINISHED!!!"
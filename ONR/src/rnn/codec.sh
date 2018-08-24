python encoder.py --model checkpoint/encoder_epoch_00000066.pth --input ./mnist.png --cuda --output ex --iterations 16
python decoder.py --model checkpoint/decoder_epoch_00000066.pth --input ./ex.npz --cuda --output ./output --iterations 16
echo " "
echo "Done, Press ENTER to end"
read
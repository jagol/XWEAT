rm results.txt;
touch results.txt;
for test_number in 11 12 13 14 15; do
    printf "test number: $test_number"
    python3 weat.py --test_number $test_number  --permutation_number 1000000 --output_file results${test_number}.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine
done

python3 weat.py --test_number 35  --permutation_number 10000 --output_file results35.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine

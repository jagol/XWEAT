rm results.txt;
touch results.txt;
for test_number in 34 35; do
    printf "test number: $test_number"
    python3 weat.py --test_number $test_number  --permutation_number 1000000 --output_file results${test_number}.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine
done

# male: german vs romanian / pl/unpl
# python3 weat.py --test_number 18  --permutation_number 1000000 --output_file results18.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine
# male: german vs arabic / pl/unpl
# python3 weat.py --test_number 11  --permutation_number 1000000 --output_file results11.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine
# male: german vs french / career/crime
# python3 weat.py --test_number 31  --permutation_number 1000000 --output_file results31.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine

# both: german vs romanian / pl/unpl
# python3 weat.py --test_number 18  --permutation_number 1000000 --output_file results18.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine
# both: german vs swiss / pl/unpl
# python3 weat.py --test_number 21  --permutation_number 1000000 --output_file results21.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine
# both: german vs swiss / career/crime
# python3 weat.py --test_number 36  --permutation_number 1000000 --output_file results36.txt --lower True --use_glove False --is_vec_format True --lang de --embeddings /home/user/jgoldz/bias/data/german_trimmed.model --similarity_type cosine

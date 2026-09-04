.PHONY: test
test:
	g++ -std=c++17 -Wall -Wpedantic -Werror -fsanitize=address -o $(NAME).o $(NAME).cpp -O3
	./$(NAME).o

speed:
	g++ -std=c++17 -Wall -Wpedantic -Werror -flto -o $(NAME).o $(NAME).cpp -O3
	./$(NAME).o

clean:
	find . -name "*.o" -type f -delete

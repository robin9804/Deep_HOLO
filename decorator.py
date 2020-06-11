# 데코레이터 사용 예제
def decorator(func):
    def decoreted():
        print("what is next")
        func()
        print("what is now?")
    return decoreted


@decorator
def main_function():
    print("this is not me")


main_function()

# 결과
#what is next
#this is not me
#what is now?
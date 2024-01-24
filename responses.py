import chatbot


def get_responses(user_input: str):
    response: str = chatbot.get_bot_response(user_input, chatbot.get_prompt_list())
    responses: list[str] = []
    if len(response) > 2000:
        res_first = response[0:len(response)//2] 
        res_second = response[len(response)//2 if len(response)%2 == 0 else ((len(response)//2)+1):] 
        responses.append(res_first)
        responses.append(res_second)
    else:
        responses.append(response)
    print(response)
    return responses
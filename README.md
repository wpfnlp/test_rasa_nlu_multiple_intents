## 如何使用Rasa NLU TensorFlow管道处理每个输入的多个意图

![](https://upload-images.jianshu.io/upload_images/4905462-0b5598e288620c23.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们介绍一个新的基于TensorFlow的Rasa NLU管，新的管道解决了聊天机器人开发人员面临的两个主要问题：
1. 你如何超越预先训练的嵌入的限制？
2. 如何构建一个可以理解多个意图的聊天机器人呢？

在这篇文章中，我们将全面了解基于TensorFlow的管道如何帮助我们解决第二个问题：多个意图。本教程的结果将是一个非常简单的聊天机器人，可以推荐聚会在柏林参加。

如果您想继续学习，可以在[此处](https://github.com/RasaHQ/tutorial-tf-pipeline)找到本教程中使用的代码和数据集。

#### 什么是新的TensorFlow管道？
处理管道是任何Rasa NLU模型的构建块。管道定义了如何解析用户输入，标记化以及如何提取功能。管道的组件很重要，因为它们对NLU模型的执行方式有直接影响。与常规Rasa NLU流水线相比，新的TensorFlow流水线可以训练为单个输入消息分配两个或更多意图的模型。例如，当用户说“Yes, make a booking. Can you also book me a taxi from the airport to the hotel?“有两个意图 - **确认应该进行预订**以及**预订出租车的额外请求**。我们可以通过分配这些输入来模拟这些输入，其中包括多个意图上面的例子将是`confirm+book_taxi`。

让我们看看它是如何在实践中完成的。

#### 创建一个聚会聊天机器人
我最近搬到了柏林，我认为加入聚会是结识该地区新人的最佳方式。这就是为什么在这个教程中我决定建立一个小型聊天机器人，可以推荐很酷的聚会在柏林参加。一点免责声明 - 出于可重复性的原因，我不打算使用任何花哨的API，但我想鼓励您使用代码，实现自定义操作，连接到实时聚会，位置或其他API并使这个聊天机器人更有趣！

1. 定义管道
让我们从本教程的全部内容 - 管道开始。下面的代码块包含我将用于聊天机器人的管道配置（请查看`config.yml`文件）。它包含一个处理参数`CountVectorsFeaturizer`，它定义了如何提取模型特征（您可以[在这里](https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#multiple-intents)阅读更多关于参数的信息）和另外一个组件`EmbeddingIntentClassifier`，它表明我们将使用TensorFlow嵌入进行意图分类。通过设置标志`intent_tokenization_flag：true`，我们告诉模型我们要将意图标签拆分为标记，这意味着模型将知道哪些意图是多意图，并且使用`intent_split_symbol`我们定义应该使用哪个字符进行拆分，在这种情况下是`+`。
```yml
# Configuration for Rasa NLU.
# https://rasa.com/docs/rasa/nlu/components/
language: en
pipeline:
  - name: "CountVectorsFeaturizer"
  - name: "EmbeddingIntentClassifier"
    intent_tokenization_flag: true
    intent_split_symbol: "+"

# Configuration for Rasa Core.
# https://rasa.com/docs/rasa/core/policies/
policies:
  - name: MemoizationPolicy
    max_history: 5
  - name: KerasPolicy
    batch_size: 50
    epochs: 200
    max_training_samples: 300
  - name: MappingPolicy
  - name: FormPolicy
  - fallback_action_name: action_default_fallback
    name: FallbackPolicy
```

2. NLU训练数据
使用TensorFlow管道的模型的训练数据是什么样的？与常规方法没有什么不同 - 唯一的补充是我们必须添加多目标输入的示例并为它们分配相应的多目标标签。下面我有一小段训练数据，我将用它来训练NLU模型（检查`data / nlu_data.md`文件）。正如您所看到的，我有一些常规示例，每个输入有一个intent，以及分配了多个Intent的示例。例如，输入“Can you suggest any cool meetups in Berlin area?”只有一个意图 - 用户要求提供聚会建议，这就是为什么它有一个分配给它的意图。另一方面，输入“Sounds good. Do you know how I could get there from home?”意味着两件事 - 用户想要加入聚会的确认和关于到达场地的交通的查询，这就是为什么这些例子具有组合的`affirm+ask_transport`意图的原因。
```Markdown
## intent: greet
- Hi
- hey
- heya
- Hello
- What's up
- Heya
- Greetings
- Good morning
- Good afternoon
- Good evening
- Hey sir
- Hi person
- Hey robot
- Hello bot

## intent: goodbye
- Bye
- Goodbye
- Talk to you later
- See you
- See you later
- Bye bye
- Bye for now
- Goodbye bot

## intent: affirm
- yes
- yup
- yes, that sounds good
- sure
- definitely
- absolutely
- please do
- yes, please
- yes for sure

## intent: deny
- no
- nope
- No, I don't think so
- Maybe not
- Not today
- No, I'm good
- No thanks

## intent: thanks
- thanks
- thank you
- Thanks a lot
- Thanks a bunch
- Thank you very much
- Thank you so much

## intent: thanks+goodbye
- Awesome. Talk to you later!
- Thanks. Bye for now
- Awesome, bye bye
- That's great. Goodbye.
- Perfect. Talk to you later
- Thanks! Goodbye
- Thank you very much. Talk to you later
- Thanks a lot. Bye for now
- Thanks bot. Goodbye

## intent: meetup
- I am new to the area. What meetups I could join in Berlin?
- I have just moved to Berlin. Can you suggest any cool meetups for me?
- I have moved to the area and would like to join some tech meetups. Any suggestions?
- I am new to London. Can you suggest any cool meetups I could attend?
- I have just arrived in Berlin. What meetups I could attend here?
- Looking for a tech meetup in the area.
- Are there any good AI meetups in Berlin?

## intent: ask_transport
- How do I get there?
- Can you tell me what is the easiest way to get to the venue?
- Tell me how should I get to the venue from home.
- Do you know how I should get to the venue of the meetup?
- Can you tell me how to get to the venue?

## intent: affirm+ask_transport
- Yes. How do I get there?
- Sounds good. Do you know how I could get there from home?
- Yes, please. How do I get there from work?
- Yes, sure! Can you also tell me what is the best way to get to the venue?
- Sure, sounds good. Can you tell the best way to get to the venue?
- Yes! Can you also recommend me the best way to get to the venue?
- Yes, sure. Also, can you tell me what is the fastest way to get to the venue?
- Sure. Can you suggests how should get there?
- Yes, thanks. I wonder, how should I get to the venue from work?
- Yes, definitely. I wonder how should I get to the venue?
- Definitely. Can you tell me how could I get to the venue of the meetup?
```
3. 训练并且测试NLU模型
一旦NLU数据准备完毕，我们就可以通过执行下面命令训练模型。
```
rasa train nlu
```
它调用Rasa NLU训练函数，提供管道配置和数据文件，并打印出训练结果。

在训练模型时，我们可以测试其在各种输入上的性能。要做到这一点：
```
rasa shell nlu
```
下面我们可以看到输入消息的模型输出“Yes. Can you give me suggestions on how to get there?”。我们可以看到，输入被归类为多意图`affirm+ ask_transport`，它基于训练数据是我们对此示例的期望。
```
Next message:
Yes. Can you give me suggestions on how to get there? 
{
  "intent": {
    "name": "affirm+ask_transport",
    "confidence": 0.9121424555778503
  },
  "entities": [],
  "intent_ranking": [
    {
      "name": "affirm+ask_transport",
      "confidence": 0.9121424555778503
    },
    {
      "name": "ask_transport",
      "confidence": 0.22044242918491364
    },
    {
      "name": "thanks+goodbye",
      "confidence": 0.022139914333820343
    },
    {
      "name": "thanks",
      "confidence": 0.0
    },
    {
      "name": "goodbye",
      "confidence": 0.0
    },
    {
      "name": "deny",
      "confidence": 0.0
    },
    {
      "name": "affirm",
      "confidence": 0.0
    },
    {
      "name": "greet",
      "confidence": 0.0
    },
    {
      "name": "meetup",
      "confidence": 0.0
    }
  ],
  "text": "Yes. Can you give me suggestions on how to get there?"
}
```

4. 定义域和训练数据
为了演示所有部分如何组合在一起，我们构建一个对话管理模型，其中包含一些模板作为响应（如前所述，为了重现性和简单性，我们不会使用任何实时API或数据库）。域文件包含模板，对话管理模型将用于响应用户（检查`domain.yml`文件）：
```yml
intents:
  - greet
  - goodbye
  - affirm
  - thanks
  - thanks+goodbye
  - meetup
  - affirm+ask_transport
  - ask_transport
  - deny


templates:
  utter_greet:
    - text: "Hey, how can I help you?"
  utter_goodbye:
    - text: "Talk to you later!"
    - text: "Goodbye :("
    - text: "Bye!"
    - text: "Have a great day!"
  utter_confirm:
    - text: "Done - I have just booked you a spot at the Bots Berlin meetup."
    - text: "Great, just made an RSVP for you."
  utter_meetup:
    - text: "Rasa Bots Berlin meetup is definitely worth checking out! They are having an event today at Behrenstraße 42. Would you like to join?"
  utter_affirm_suggest_transport:
    - text: "Great, I have just booked a spot for you. The venue is close to the Berlin Friedrichstraße station, you can get there by catching U-Bahn U6."
  utter_suggest_transport:
    - text: "The venue is close to the Berlin Friedrichstraße station, so the best option is to catch a U-Bahn U6."
  utter_thanks:
    - text: "You are very welcome."
    - text: "Glad I could help!"
  utter_deny:
    - text: "That's a shame. Let me know if you change your mind."


actions:
  - utter_greet
  - utter_goodbye
  - utter_confirm
  - utter_meetup
  - utter_affirm_suggest_transport
  - utter_suggest_transport
  - utter_thanks
  - utter_deny
```
这些模板将用作对用户输入的响应，具体取决于它们在创建故事数据时的使用方式。我们将在下一节中更详细地研究它。

在继续之前，我想指出像`utter_goodbye`这样的模板有多个可能的响应。添加这样的选项是使聊天机器人更有趣并防止它在每次对话中重复相同答案的好方法。

5. 生成故事集
像往常一样，为了训练对话管理模型，我们需要一些故事。新的TensorFlow管道不需要故事数据的任何特殊格式 - 我们可以使用先前定义的多个或单个意图和相应的操作。在下面的表格中，您可以找到两个非常相似的故事，将用于我们的模型 - 一个具有多个意图，另一个具有单个意图（检查`data / stories.md`文件）
![](https://upload-images.jianshu.io/upload_images/4905462-49c579e6be9f0dd8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

第一个故事有两个多重意图 - `affirm+ ask_transport`，对应于一个用户说“Yes, book me a spot at the meetup. Also, can you tell me how should I get to the venue?”和另一个多意图的`thanks+goodbye`，对应于用户说“Thank you. Talk to you later”。第二个故事代表了一个非常相似的对话，但它只使用单个意图。与第二个故事相比，第一个故事反映了更有机和人类的对话。

需要强调的另一件事是，有多种不同方式可以编写具有多种意图的故事。下表显示了同一对话的不同表示：

![](https://upload-images.jianshu.io/upload_images/4905462-52af018d82e9e586.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

故事**find_meetup_01**使用特殊操作`utter_affirm_suggest_transport`作为对多意图`affirm+ ask_transport`的响应。或者，就像在**find_meetup_03**中一样，我们可以使用两个单独的模板 - `utter_confirm`和`utter_suggest_transport`来编写这个故事，它们也可以用作对单意图输入的响应。关于具有多意图的故事的另一个重要注意事项是没有必要为每一个多意图执行动作。例如，故事**find_meetup_03**有两个动作作为对多意图`thanks+goodbye`的响应，但是，就像在故事**find_meetup_04**中一样，跳过其中一个标记的动作是完全可以的。

决定哪种方法最好高度依赖于聊天机器人的域和逻辑 - **在某些情况下，为多方意图创建单独的操作绝对没有必要，您可以使用所有相同的操作作为多方意图的响应和对于单意图。**在构建聊天机器人之前，确定哪些操作确实需要与聊天机器人进行自然对话始终是一种很好的做法。

6. 训练对话管理模型并测试机器人
为了训练模型我们可以使用下面的命令去训练NLU和Core模型，并且保存成一个`tar.gz`的文件。你可以在训练时`config.yml`文件中指定参数像`epochs的数目。
```
rasa train
```
一旦训练结束，就到了最激动人心的时刻-测试你的机器人。我们可以在命令行中使用下面代码块去启动它，它将加载对话管理模型和NLU模型并在命令行中启动它。
```
rasa shell
```
这是与聊天机器人的实际对话：
```
Bot loaded. Type a message and press enter (use '/stop' to exit): 
Your input ->  Hello                                                            
Hey, how can I help you?
Your input ->  I am new to Berlin and I would like to join some meetups. Any sug
gestions?                                                                       
Rasa Bots Berlin meetup is definitely worth checking out! They are having an event today at Behrenstraße 42. Would you like to join?
Your input ->  Sure! Can you also tell me how can I get to the venue?           
Done - I have just booked you a spot at the Bots Berlin meetup.
The venue is close to the Berlin Friedrichstraße station, so the best option is to catch a U-Bahn U6.
Your input ->  Thanks a lot. Talk to you later.                                 
You are very welcome.
Talk to you later!
Your input ->  /stop                                                            
2019-09-19 16:13:33 INFO     root  - Killing Sanic server now.
```
![运行截图](https://upload-images.jianshu.io/upload_images/4905462-5fd7bbac5f71a461.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
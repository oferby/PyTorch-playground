from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

modelname = 'deepset/bert-base-cased-squad2'

model = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

context = ['Keith recently came back from a trip to Chicago, Illinois. ' \
           'This midwestern metropolis is found along the shore of Lake Michigan. ' \
           'During his visit, Keith spent a lot of time exploring the city to visit important landmarks and monuments. \
           Keith loves baseball, and he made sure to take a visit to Wrigley Field. ' \
           'Not only did he take a tour of this spectacular stadium, but he also got to watch a Chicago Cubs game. ' \
           'In the stadium, Keith and the other fans cheered for the Cubs. Keith was happy that the Cubs won with a score of 5-4. \
           Chicago has many historic places to visit. Keith found the Chicago Water Tower impressive as it is one ' \
           'of the few remaining landmarks to have survived the Great Chicago Fire of 1871. ' \
           'Keith also took a walk through Jackson Park, a great outdoor space that hosted the World’s Fair of 1892. ' \
           'The park is great for a leisurely stroll, and it still features some of the original architecture and ' \
           'replicas of monuments that were featured in the World’s Fair. \
           During the last part of his visit, Keith managed to climb the stairs inside of the Willis Tower, ' \
           'a 110-story skyscraper. Despite the challenge of climbing the many flights of stairs, ' \
           'Keith felt that reaching the top was worth the effort. From the rooftop, Keith received a gorgeous view ' \
           'of the city’s skyline with Lake Michigan in the background.',
           'In December, Beyoncé along with a variety of other celebrities teamed up and produced a video campaign ' \
           'for "Demand A Plan", a bipartisan effort by a group of 950 US mayors and others designed to influence ' \
           'the federal government into rethinking its gun control laws, following the Sandy Hook Elementary School shooting. ' \
           'Beyoncé became an ambassador for the 2012 World Humanitarian Day campaign donating her song ' \
           '"I Was Here" and its music video, shot in the UN, to the campaign. In 2013, it was announced that Beyoncé would ' \
           'work with Salma Hayek and Frida Giannini on a Gucci "Chime for Change" campaign that aims to spread female ' \
           'empowerment. The campaign, which aired on February 28, was set to her new music. ' \
           'A concert for the cause took place on June 1, 2013 in London and included other acts like Ellie Goulding, ' \
           'Florence and the Machine, and Rita Ora. In advance of the concert, she appeared in a campaign video released on ' \
           '15 May 2013, where she, along with Cameron Diaz, John Legend and Kylie Minogue, described inspiration from ' \
           'their mothers, while a number of other artists celebrated personal inspiration from other women, leading to ' \
           'a call for submission of photos of women of viewers\' inspiration from which a selection was shown at ' \
           'the concert. Beyoncé said about her mother Tina Knowles that her gift was "finding the best qualities in ' \
           'every human being." With help of the crowdfunding platform Catapult, visitors of the concert could choose ' \
           'between several projects promoting education of women and girls. Beyoncé is also taking part in "Miss a Meal", ' \
           'a food-donation campaign, and supporting Goodwill charity through online charity auctions at Charitybuzz that support ' \
           'job creation throughout Europe and the U.S.'
           ]

questions = [
    "Where is Chicago in the United States?",
    "When did Beyonce start becoming popular?"
    "What sport do the Chicago Cubs play in Wrigley Field?",
    "Why was Keith impressed by the Chicago Water Tower?",
    "What event was important for Chicago in 1892?",
    "How did Keith arrive to the rooftop of the Willis Tower?"
]

vec = tokenizer.encode(questions[0], truncation=True, padding=True)
print(vec)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

ret = nlp({
    'question': questions[0],
    'context': context[0]
})
print(ret)

ret = nlp({
    'question': questions[0],
    'context': context[1]
})
print(ret)

ret = nlp({
    'question': questions[1],
    'context': context[1]
})
print(ret)

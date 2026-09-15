# Selected assistant-response cases

These seven alpha075 answers illustrate [Technical Report §8.5](../TECHNICAL_REPORT.md#85-qualitative-cases-successes-and-failures).
They were selected to explain complete success, format/interface success with
content failure, and an incorrect calling interface. They are not a random sample
or a basis for estimating performance.

Original labels below are copied from `assistant_review_fable_v1`; the separate
owner-clarification column applies the four amendments in [C-ASSIST-05](../provenance/PUBLIC_EVIDENCE_INDEX.json).
**One of these seven cases changed:** `R1DEV-FOLLOWUP-007` went from content/joint
false to true. The other six cases are unchanged. Full original prompt
messages and stored responses are preserved, including errors and repetition.
Earlier assistant turns in the prompts are fixed supplied context. This edition
ran no new model inference or generated-code execution and did not revise scores.
Editorial explanations are separated from the original review reasons.

[Machine-readable cases](ASSISTANT_CASE_STUDIES.json) include task and review IDs,
source-member hashes, output hashes and labels. The full evaluation collection
remains outside this repository. Source labels are retained where the frozen task
records them; component attribution and documentation scope are described in
[SOURCE_NOTICE.md](../SOURCE_NOTICE.md) and [DOCUMENTATION_LICENSE.md](../DOCUMENTATION_LICENSE.md).

## Index

| Case | Task | Original review outcome | Owner-clarification content / joint |
|---|---|---|---|
| [PRACTICAL_077_M2](#practical_077_m2) | One-sentence summary: complete pass | content `True`, format `True`, interface `None` | `True` / `True` (unchanged) |
| [R1DEV-REWRITE-008](#r1dev-rewrite-008) | One-sentence summary: format pass, content failure | content `False`, format `True`, interface `None` | `False` / `False` (unchanged) |
| [R1DEV-FOLLOWUP-001](#r1dev-followup-001) | Context update: complete pass | content `True`, format `None`, interface `None` | `True` / `True` (unchanged) |
| [R1DEV-FOLLOWUP-007](#r1dev-followup-007) | Context update: judgment changes between review versions | content `False`, format `None`, interface `None` | `True` / `True` (amended) |
| [PYTHON_041_M1](#python_041_m1) | Python: correct matching operation, wrong return type | content `False`, format `True`, interface `True` | `False` / `False` (unchanged) |
| [R1DEV-PYTHON-013](#r1dev-python-013) | Python: correct non-empty mean, missing empty-list branch | content `False`, format `None`, interface `True` | `False` / `False` (unchanged) |
| [PYTHON_035_M1](#python_035_m1) | Python: calling interface and implementation both fail | content `False`, format `True`, interface `False` | `False` / `False` (unchanged) |

A null format/interface result means that axis is inapplicable here; the records retain applicability flags.

## PRACTICAL_077_M2

**One-sentence summary: complete pass**

The response keeps the proposed phonics workshop and practical activities in one sentence, without inventing approval.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**system**

```text
Extract and present the main key point of the input text in one very short sentence, including essential details like dates or locations if necessary.
```

**user**

```text
Hi Rachel,

I hope you're doing well! As we prepare for our upcoming presentation at the education conference, I wanted to reach out and brainstorm some ideas for our workshop. I was thinking we could focus on the importance of phonics instruction in early literacy development and share some practical strategies that we've found effective in our classrooms.

I'd love to hear your thoughts on this topic and any specific activities or resources you think would be valuable to share with our audience. Maybe we could also include a hands-on component where participants get to try out some of the strategies we discuss?

Let me know what you think and if you have any other ideas for our workshop. I'm excited to collaborate with you on this!

Best regards,
Emily
```

### Stored alpha075 answer

```text
Emily proposes a workshop on phonics instruction and suggests hands-on activities and resources.
```

</details>

**Historical review reason:** One sentence stating that Emily proposes a phonics-instruction workshop with hands-on activities and resources; no approval or topic change asserted.

**Required content in the frozen task:**

- Emily提出教育会议的phonics/自然拼读教学工作坊。

Review ID: `FA-c41ee72d6f1c102a87a5`. Source family: `smol-summarize-20k`.

## R1DEV-REWRITE-008

**One-sentence summary: format pass, content failure**

The requested action is to finish drilling by noon on weekdays. The response instead reports completion by 1:30 pm; it loses the request and changes the relevant time.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**system**

```text
Extract and present the main key point of the input text in one very short sentence, including essential details like dates or locations if necessary.
```

**user**

```text
Uma. I need you to finish the drilling by 12 pm on weekdays. It carried on until 1:30 pm on tuesday and again on wednesday. I'm happy to be flexible on saturdays. Reply by wednesday. Victor
```

### Stored alpha075 answer

```text
By 1:30 pm on weekdays, the drilling was completed.
```

</details>

**Historical review reason:** One sentence, but it states the opposite of the text ('the drilling was completed' by 1:30 pm) and omits that Victor asks Uma to finish by 12 pm on weekdays.

**Required content in the frozen task:**

- one sentence stating that Victor asks Uma to finish the drilling by 12 pm on weekdays
- at least one essential detail: the late days (Tuesday/Wednesday) or the reply deadline (Wednesday)

Review ID: `FA-2e7f934521a13a8d2509`. Source family: `project-authored development task`.

## R1DEV-FOLLOWUP-001

**Context update: complete pass**

The fee changes to $240 while the $60 deposit stays fixed. The response gives the updated $180 balance. The earlier assistant turn is supplied context, not a new free-running model answer.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**user**

```text
Dear Lena,

Thank you for enrolling in Bread Baking. The course runs on Friday evenings at 7:30 pm for 6 weeks, starting on 11 October. The fee is $180; a deposit of $60 is due by 2 October and the balance on the first evening. Please note that all materials are included. The course only runs if at least 8 people enrol. The deposit is non-refundable.

Kind regards,
Adult Learning Office

How much do I still owe on the first evening?
```

**assistant**

```text
$120, on the first evening (11 October): the fee is $180 and the $60 deposit is due by 2 October.
```

**user**

```text
Correction from the office: the fee is actually $240, deposit unchanged. What's the balance now?
```

### Stored alpha075 answer

```text
The balance on the first evening is $180.
```

</details>

**Historical review reason:** Gives the corrected balance of $180 ($240 minus the $60 deposit).

**Required content in the frozen task:**

- new balance on the first evening is $180 (corrected fee $240 minus the unchanged $60 deposit)

Review ID: `FA-a37a64810b132fce3e1e`. Source family: `project-authored development task`.

## R1DEV-FOLLOWUP-007

**Context update: judgment changes between review versions**

The response states the new starting time and one-hour duration without spelling out the end time. The original model-assisted review marked it false; the owner-clarification version accepts the unchanged answer as true. This illustrates a scoring boundary, rather than a stable failure across versions.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**user**

```text
Kingsway Health Clinic: Monday to Friday 8:30 am to 6 pm, closed for lunch between 1 pm and 2 pm. Saturday 9 am to 1 pm. Closed on Sundays. Walk-in patients are seen in the mornings only. The clinic will be closed on Wednesday 19 October for a staff training day.

When is the lunch closure?
```

**assistant**

```text
Between 1 pm and 2 pm on weekdays, so 1 hour.
```

**user**

```text
The receptionist says the lunch break now starts at 1:30 pm and still lasts an hour. When is the closure now?
```

### Stored alpha075 answer

```text
On weekdays, the receptionist says the lunch break now starts at 1:30 pm and still lasts an hour.
```

</details>

**Historical review reason:** Restates the receptionist's sentence without computing the new closure interval; the required 1:30 pm to 2:30 pm is not stated.

**Required content in the frozen task:**

- new lunch closure is 1:30 pm to 2:30 pm (start moved 30 min later, still one hour)

Review ID: `FA-d113b756835afbd0a234`. Source family: `project-authored development task`.

## PYTHON_041_M1

**Python: correct matching operation, wrong return type**

The name, parameters and use of re.findall match the request, but the function returns matches instead of their count.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**user**

```text
Create a Python function `count_matches_regex` that takes a string `text` and a regular expression pattern `pattern` and returns the number of matches found in the string. Use the `re` module for regular expression matching.
```

### Stored alpha075 answer

````text
Here's how you can implement this:

```python
import re

def count_matches_regex(text, pattern):
    matches = re.findall(pattern, text)
    return matches
```
````

</details>

**Historical review reason:** Name and (text, pattern) signature match and re is used, but the function returns the list from re.findall instead of its length; the requested count is never produced.

**Required content in the frozen task:**

- 通过所附原契约有限功能测试及规定返回类型；测试通过不证明所有可能输入。

Review ID: `FA-ff71cc6bbf1e08e278e8`. Source family: `self-oss-instruct`.

## R1DEV-PYTHON-013

**Python: correct non-empty mean, missing empty-list branch**

The expression computes the mean of ordinary non-empty numeric lists. For an empty list it divides by zero, never using default. The explanatory text incorrectly claims that default is returned.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**user**

```text
Write a Python function `average_or_default(numbers, default)` returning the arithmetic mean of the list, or the caller-supplied `default` value unchanged when the list is empty.
```

### Stored alpha075 answer

````text
Here's how you can implement this function:

```python
def average_or_default(numbers, default):
    return sum(numbers) / len(numbers)
```

The `default` value is returned when the list is empty. If the list is empty, the `average_or_default` function returns the default value. Otherwise, it returns the sum of the numbers.
````

</details>

**Historical review reason:** Never checks for an empty list; sum([])/len([]) raises ZeroDivisionError and default is never returned. The prose falsely claims the default is returned.

**Required content in the frozen task:**



Review ID: `FA-b4b4ff959d520be8df0f`. Source family: `project-authored development task`.

## PYTHON_035_M1

**Python: calling interface and implementation both fail**

The supplied assertions call a zero-argument function. The response requires a lines argument and returns a single formatted fragment rather than 100 numbered lines. The complete stored answer below retains its repetitive tail.

<details>
<summary>Complete original prompt and answer</summary>

### Prompt messages

**user**

````text
Write a Python function to generate the following string with 100 lines, each consisting of line number and the word "Python":

```
1 Python
2 Python
3 Python
4 Python
5 Python
...
```

Use string formatting to achieve this.

Your code should pass the following assertions:
```python
assert generate_string_with_100_lines().startswith("1 Python\n")
assert generate_string_with_100_lines().endswith("100 Python\n")
assert generate_string_with_100_lines().count("\n") == 100
assert generate_string_with_100_lines().count("Python") == 100
```
````

### Stored alpha075 answer

````text
Here's a Python function that generates the given string with 100 lines:

```python
def generate_string_with_100_lines(lines):
    return f"python: {lines}"
```

The function uses string formatting to generate the string. The word "Python" is formatted as "Python", and the word "Python" is formatted as "Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python Python
````

</details>

**Historical review reason:** Defines generate_string_with_100_lines(lines) with a required argument although the assertions call it with none, and returns f"python: {lines}", not 100 numbered lines.

**Required content in the frozen task:**

- 通过所附原契约有限功能测试及规定返回类型；测试通过不证明所有可能输入。

Review ID: `FA-d4747e4b8b36756fd9c0`. Source family: `self-oss-instruct`.

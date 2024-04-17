
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
import pandas as pd
import numpy as np
import datetime
from typing import Iterator


def LLM(values):

    openai_api_key = "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"
    llm = OpenAI(api_key=openai_api_key)
    # variables = ['current_balance', 'current_month_debit', 'previous_month_debit', 'current_month_balance', 'average_monthly_balance_prevQ', 'previous_month_balance', 'previous_month_end_balance', 'average_monthly_balance_prevQ2', 'days_since_last_transaction', 'previous_month_credit', 'churn_prediction']

    # for key in values.keys():
    #   if key not in variables :
    #     del values[key]

    template = """You are a bank manager inspecting if a customer will churn/not churn in the near future. You have a data science model that is making predictions.
              Analyze the statistical measures of the variables from the training data and compare it with the variable values for the customer and create bullet points for the manager to understand why this prediction was made.
              Below are the statistical measures of the features from training data:
              1. Variable : current_balance, Mean : 7552.9258603283115 , Median : 3325.03, 25th Percentile : 1767.29, 75th Percentile : 6810.205
              2. Variable : current_month_debit, Mean : 4076.7408327040025 , Median : 182.3, 25th Percentile : 0.46, 75th Percentile : 1526.4099999999999
              3. Variable : previous_month_debit, Mean : 3725.2528192694467 , Median : 194.86, 25th Percentile : 0.47, 75th Percentile : 1557.2150000000001
              4. Variable : current_month_balance, Mean : 7624.336430700743 , Median : 3503.78, 25th Percentile : 2010.0149999999999, 75th Percentile : 6864.77
              5. Variable : average_monthly_balance_prevQ, Mean : 7660.709340593824 , Median : 3601.46, 25th Percentile : 2198.7200000000003, 75th Percentile : 6821.84
              6. Variable : previous_month_balance, Mean : 7654.748472912277 , Median : 3514.47, 25th Percentile : 2081.8199999999997, 75th Percentile : 6787.275
              7. Variable : previous_month_end_balance, Mean : 7661.40589252355 , Median : 3419.71, 25th Percentile : 1898.3049999999998, 75th Percentile : 6828.525
              8. Variable : average_monthly_balance_prevQ2, Mean : 7222.216939465004 , Median : 3368.14, 25th Percentile : 1797.945, 75th Percentile : 6617.585
              9. Variable : days_since_last_transaction, Mean : 167.17202591517946 , Median : 127.0, 25th Percentile : 108.0, 75th Percentile : 192.0
              10. Variable : previous_month_credit, Mean : 3679.4894650025835 , Median : 0.98, 25th Percentile : 0.36, 75th Percentile : 1048.52

              Below are the variable values for the customer under inspection :
              1. Variable : current_balance, Value : {current_balance}
              2. Variable : current_month_debit, Value : {current_month_debit}
              3. Variable : previous_month_debit, Value : {previous_month_debit}
              4. Variable : current_month_balance, Value : {current_month_balance}
              5. Variable : average_monthly_balance_prevQ, Value : {average_monthly_balance_prevQ}
              6. Variable : previous_month_balance, Value : {previous_month_balance}
              7. Variable : previous_month_end_balance, Value : {previous_month_end_balance}
              8. Variable : average_monthly_balance_prevQ2, Value : {average_monthly_balance_prevQ2}
              9. Variable : days_since_last_transaction, Value : {days_since_last_transaction}
              10. Variable : previous_month_credit, Value : {previous_month_credit}

              The predicted value for this customer is : {churn_prediction}
              0 means the customer will not churn. 1 means the customer will churn
              Your task : Provide precise bullet points which will help the manager understand the reasons for why the customer will churn or not by comparing the customers values for the variables to the population statistics. Do not include \n or \t. The output should directly be presentable as print. Please do not leave empty responses and only give 5 points"""

    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(values)

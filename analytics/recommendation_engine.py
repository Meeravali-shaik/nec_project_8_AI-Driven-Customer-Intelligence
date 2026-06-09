import pandas as pd

def get_recommendations(customer):

    category = customer["Product_Category"]

    recommendations = {

        "Electronics": [
            "Laptop",
            "Smartwatch",
            "Earbuds"
        ],

        "Insurance": [
            "Life Insurance",
            "Health Insurance"
        ],

        "Travel": [
            "Holiday Package",
            "Flight Ticket"
        ],

        "Education": [
            "AI Course",
            "Data Analytics Course"
        ]
    }

    return recommendations.get(
        category,
        ["General Product"]
    )
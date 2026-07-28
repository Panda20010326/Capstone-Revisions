from typing import Any

def generate_explanation(user_profile: dict[str, Any], job: dict[str, Any],
                         predictions: dict[str, Any],
                         score_breakdown: dict[str, float],
                         matched_skills: list[str]) -> list[str]:
    reasons = []
    occupation = predictions.get("predicted_occupation")
    income = predictions.get("predicted_income")
    employment_probability = predictions.get("employment_probability")

    if score_breakdown.get("occupation", 0) >= 0.70 and occupation:
        reasons.append(f"The role aligns with your predicted occupation area, {occupation}.")
    if matched_skills:
        reasons.append("The job description includes skills that match your profile: "
                       + ", ".join(matched_skills[:5]) + ".")
    if score_breakdown.get("salary", 0) >= 0.70 and income:
        reasons.append(f"The advertised salary is reasonably close to your predicted annual income of ${float(income):,.0f}.")
    if score_breakdown.get("location", 0) >= 0.90 and user_profile.get("preferred_city"):
        reasons.append(f"The job is located in your preferred area, {user_profile['preferred_city']}.")
    if score_breakdown.get("contract", 0) >= 0.90:
        reasons.append("The employment type matches your stated work preference.")
    if score_breakdown.get("profile_fit", 0) >= 0.70:
        reasons.append("The role has a strong fit with the professional profile learned by the ProfileEncoder.")
    if employment_probability is not None and float(employment_probability) >= 0.70:
        reasons.append(f"The employment model estimated a strong employment probability of {float(employment_probability):.0%}.")
    if not reasons:
        reasons.append("This role was selected because it achieved one of the highest overall scores among the available job listings.")
    return reasons

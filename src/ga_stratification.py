import pandas as pd


def build_balanced_ga_bins(gestational_age, y, start_bins=4, max_iter=20):
    """
    Create quantile-based gestational-age bins.

    If a bin contains only one class, it is merged with a neighboring bin.
    This helps preserve class balance inside GA groups.
    """
    bins = pd.qcut(gestational_age, q=start_bins, duplicates="drop")
    codes = bins.cat.codes.copy()

    def group_has_both_classes(group_code, contingency_table):
        if group_code not in contingency_table.index:
            return False

        n_normal = int(contingency_table.loc[group_code][0]) if 0 in contingency_table.columns else 0
        n_iugr = int(contingency_table.loc[group_code][1]) if 1 in contingency_table.columns else 0

        return n_normal > 0 and n_iugr > 0

    for _ in range(max_iter):
        contingency_table = pd.crosstab(codes, y)
        ordered_groups = sorted(codes.unique())

        bad_groups = [
            group_code
            for group_code in ordered_groups
            if not group_has_both_classes(group_code, contingency_table)
        ]

        if not bad_groups:
            break

        group_to_merge = bad_groups[0]
        group_index = ordered_groups.index(group_to_merge)

        if group_index == 0:
            target_group = ordered_groups[group_index + 1]
        elif group_index == len(ordered_groups) - 1:
            target_group = ordered_groups[group_index - 1]
        else:
            left_group = ordered_groups[group_index - 1]
            right_group = ordered_groups[group_index + 1]

            left_size = int((codes == left_group).sum())
            right_size = int((codes == right_group).sum())

            target_group = left_group if left_size >= right_size else right_group

        codes = codes.where(codes != group_to_merge, target_group)

        unique_groups = sorted(codes.unique())
        codes = codes.map({old: new for new, old in enumerate(unique_groups)})

    return codes.astype(int)

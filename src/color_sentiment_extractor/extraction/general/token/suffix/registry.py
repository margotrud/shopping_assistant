from extraction.general.token.suffix.recovery import (recover_y, recover_ed,
                                                      recover_ee_to_y,
                                                      recover_ing,
                                                      recover_ish,
                                                      recover_ly,
                                                      recover_en,
                                                      recover_ness,
                                                      recover_ier,
                                                      recover_er,
                                                      recover_ied
                                                      )

SUFFIX_RECOVERY_FUNCS = [
    recover_y,
    recover_ed,
    recover_ee_to_y,  # ← inserted here
    recover_ing,
    recover_ish,
    recover_ly,
    recover_en,
    recover_ness,
    recover_ier,
    recover_er,
    recover_ied,
]
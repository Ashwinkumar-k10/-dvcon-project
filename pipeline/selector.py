import config

class TaskSelector:
    def __init__(self, detector, scorer):
        self.detector = detector
        self.scorer   = scorer

    def select(self, image_path, task_name):
        task_info = config.TASK_DEFINITIONS.get(task_name, {})
        primary   = set(task_info.get("primary",   []))
        secondary = set(task_info.get("secondary", []))

        all_dets = self.detector.detect(image_path)
        if not all_dets:
            return {
                "status"    : "no objects detected",
                "task"      : task_name
            }

        scored = self.scorer.score(image_path, all_dets, task_name)

        pm = [d for d in scored if d["class_name"] in primary]
        sm = [d for d in scored if d["class_name"] in secondary]

        if pm:    best, mt = pm[0], "primary"
        elif sm:  best, mt = sm[0], "secondary"
        else:     best, mt = scored[0], "clip_fallback"

        return {
            "task"       : task_name,
            "status"     : "success",
            "match_type" : mt,
            "selected"   : best,
            "all_scored" : scored,
        }
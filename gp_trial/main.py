from bo_session import BOSession, LOG_PATH, used_structure_ids


def main():
    session = BOSession()

    print("=== warm start ===")
    print(session.status_line())

    print("\n=== BAYESIAN OPTIMIZATION LOOP ===")
    while True:
        done, reason = session.is_finished()
        if done:
            print(reason)
            break

        params = session.next_params()
        print(f"\nExperiment {session.experiment_num + 1}")
        print(f"Amplitude: {params[0]:.2f}, Rise: {params[1]:.2f}, Dwell: {params[2]:.2f}, Fall: {params[3]:.2f}")

        structure_id = input("Physical structure ID: ").strip()
        if structure_id in used_structure_ids(LOG_PATH):
            confirm = input(f"Structure '{structure_id}' is already logged -- continue anyway? [y/N]: ").strip().lower()
            if confirm != "y":
                continue

        while True:
            image_path = input("Enter path to microscope image: ").strip().strip('"\'')
            result = session.submit_result(structure_id, image_path)
            if result is not None:
                break
            print("No numeric duty cycle could be computed from that image. Enter a different image path.")

        mean, std, output, diagnostics = result
        print(f"Duty cycle: {mean:.2f}% ± {std:.2f}% | Output: {output:.4f}")
        if diagnostics["classification"] == "spatially_varying":
            print(
                f"WARNING: sample varies spatially -- {diagnostics['problem_fraction']:.0%} of the ROI measured as "
                f"overpoled/unstable rather than clean. Duty cycle/std above reflect ONLY the clean region; "
                f"the logged output already includes a penalty for this."
            )

    best = session.best_result()
    if best is not None:
        best_params, best_output = best
        print("\n=== BEST FOUND (from real experiments) ===")
        print(f"Amplitude: {best_params[0]:.2f}, Rise: {best_params[1]:.2f}, "
              f"Dwell: {best_params[2]:.2f}, Fall: {best_params[3]:.2f}")
        print(f"Output: {best_output:.4f}")
    else:
        print("\nNo real experiments were run.")


if __name__ == "__main__":
    main()

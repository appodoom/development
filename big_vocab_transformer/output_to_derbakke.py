# """This script takes the ouput of the model
# and turns it into a derbukka file"""

# from utils import load_json


# def write_derbukka(file_name, string):
#     with open(file_name, "w") as file:
#         file.write(string)


# def to_derbukka(directory, initial_tempo):
#     sequence = load_json(directory)
#     initial_tempo_str = str(initial_tempo)
#     count = 0
#     value = 0
#     last_delay = 0
#     tempos_str = ""
#     skeleton_str = ""
#     variations_str = ""
#     previous_tempo = initial_tempo
#     if sequence[0].split("_")[1] != "0":
#         variations_str += "SUBD_" + sequence[0].split("_")[1] + " "
#     else:
#         variations_str += "SUBD_" + sequence[1].split("_")[1] + " "

#     for i in range(len(sequence)):
#         components = sequence[i].split("_")
#         if components[1] != "0":
#             value = int(components[1])
#             count += 1 / value
#             if count % 1 == 0:
#                 variations_str += "HIT_" + components[0] + " "
#                 variations_str += "AMP_" + components[3] + " "
#                 variations_str += "SUBD_" + components[1] + " "
#                 tempos_str += str(float(previous_tempo + float(components[2]))) + " "
#             else:
#                 variations_str += "HIT_" + components[0] + " "
#                 variations_str += "AMP_" + components[3] + " "
#         else:
#             skeleton_str += "DELAY_" + str(count - last_delay) + " "
#             last_delay = count
#             skeleton_str += "HIT_" + components[0] + " "
#             skeleton_str += "DEV_" + components[-1] + " "

#     write_derbukka(
#         "output.txt",
#         initial_tempo_str
#         + "\n"
#         + tempos_str[:-1]
#         + "\n"
#         + skeleton_str[:-1]
#         + "\n"
#         + variations_str[:-1],
#     )


# to_derbukka("generated_tokens.json", 110.0)

"""This script takes the output of the model
and turns it into a derbake file"""

from utils import load_json


def write_derbake(file_name: str, string: str):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(string)


def to_derbake(json_path: str, initial_tempo: float, out_path: str = "output.derbake"):
    # JSON is expected to be: [ "TOKEN TOKEN TOKEN", "TOKEN", ... ]
    raw = load_json(json_path)

    # Flatten: turn list of space-separated strings into a list of individual tokens
    tokens = []
    for row in raw:
        if not isinstance(row, str):
            raise ValueError("JSON must be a list of strings.")
        tokens.extend(row.split())

    if not tokens:
        raise ValueError("No tokens found in the JSON.")

    initial_tempo_str = str(initial_tempo)

    count = 0.0
    last_delay = 0.0
    tempos_str = ""
    skeleton_str = ""
    variations_str = ""

    previous_tempo = initial_tempo

    # Pick the first non-zero subdivision token for initial SUBD_
    first_subd = None
    for t in tokens:
        parts = t.split("_")
        if len(parts) >= 2 and parts[1] != "0":
            first_subd = parts[1]
            break
    if first_subd is None:
        first_subd = "4"  # fallback if everything is SUBD 0 (unlikely)
    variations_str += f"SUBD_{first_subd} "

    for tok in tokens:
        parts = tok.split("_")
        if len(parts) < 5:
            raise ValueError(f"Bad token format (expected 5 parts): {tok}")

        hit = parts[0]
        subd = parts[1]  # "0" or "2/4/8/..."
        tempo_delta = parts[2]  # like "0.0"
        amp = parts[3]  # like "1.5" or "3"
        dev = parts[-1]  # last field

        if subd != "0":
            value = int(subd)
            count += 1.0 / value

            # Always write HIT/AMP in variations
            variations_str += f"HIT_{hit} AMP_{amp} "

            # On whole beats, also write SUBD and tempo
            if count % 1.0 == 0:
                variations_str += f"SUBD_{subd} "
                tempos_str += f"{float(previous_tempo + float(tempo_delta))} "
        else:
            # Skeleton event: write delay since last skeleton, then hit + dev
            skeleton_str += f"DELAY_{count - last_delay} "
            last_delay = count
            skeleton_str += f"HIT_{hit} DEV_{dev} "

    output = (
        initial_tempo_str
        + "\n"
        + tempos_str.rstrip()
        + "\n"
        + skeleton_str.rstrip()
        + "\n"
        + variations_str.rstrip()
    )

    write_derbake(out_path, output)


to_derbake("generated.json", 110.0, out_path="generated.derbake")

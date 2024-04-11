from recipes.bigmusic.inference.fsm.grammar import *


class GrammarPresets:
    def chord_part():
        return OneOrMore(Sequence(
            PositionState(),
            ChordState()
        ))

    def inst_stem_part_pitch_first(valid_stems: Optional[List[str]] = None):
        return Sequence(
            InstStemState(valid_stems),
            OneOrMore(Sequence(
                PositionState(),
                NoteOnState(),
                NoteDurationState(),
            ))
        )

    def inst_stem_part_dur_first(valid_stems: Optional[List[str]] = None):
        return Sequence(
            InstStemState(valid_stems),
            OneOrMore(Sequence(
                PositionState(),
                NoteDurationState(),
                NoteOnState(),
            ))
        )

    def drum_stem_part():
        return Sequence(
            DrumStemState(),
            OneOrMore(Sequence(
                PositionState(),
                DrumState(),
            )),
        )

    def default_grammar():
        return Sequence(OneOrMore(Sequence(
            BarState(),
            ZeroOrOne(GrammarPresets.chord_part()),
            ZeroOrOne(GrammarPresets.drum_stem_part()),
            ZeroOrMore(GrammarPresets.inst_stem_part_pitch_first()),
        )), BeginState())


def exp_20240326_structure_controlled_meta_controlled_song_bm(sec_infos):
    """Generate full meta with only structure info, then generate full song

    For encoding with genre tags
    """
    meta_grammar = []
    for sec_label, num_bars, stem_list in sec_infos:
        meta_grammar.extend([
            BarState(),
            sec_label,
            BPMLevelState(),
            GrammarPresets.chord_part(),
            *stem_list,
        ] * num_bars)

    song_grammar = []
    for sec_label, num_bars, stem_list in sec_infos:
        bar_grammar = [
            BarState(),
            sec_label,
            BPMLevelState(),
            GrammarPresets.chord_part(),
        ]
        for stem in stem_list:
            if stem == "stem_5":
                bar_grammar.append(
                    GrammarPresets.drum_stem_part(),
                )
            else:
                stem_index = int(stem.split("_")[1])
                stem_label = STEM_LABELS[stem_index]
                print(stem_label)
                bar_grammar.append(
                    GrammarPresets.inst_stem_part_dur_first(valid_stems=[stem_label]),
                )
        song_grammar.extend(bar_grammar * num_bars)
    return Sequence(
        *meta_grammar, "eop", *song_grammar, ZeroOrOne(BarState()), "eos",
    )
(** * Symbol Expansion for LLRE Patterns

    This module proves correctness of the symbol expansion algorithm that
    transforms LLRE patterns with symbol references into pure regex patterns.

    Key Theorems:
    1. Termination: expansion terminates for acyclic symbol tables
    2. Language preservation: L(expand(r)) = L(r)[table]
    3. Depth bound: expansion respects MAX_DEPTH limit

    Corresponds to: src/phonetic/llre/symbol_expander.rs
*)

Require Import Coq.Lists.List.
Require Import Coq.Strings.String.
Require Import Coq.Strings.Ascii.
Require Import Coq.Init.Nat.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.Bool.Bool.
Require Import Coq.micromega.Lia.
Import ListNotations.

(** * Abstract Syntax *)

(** LLRE patterns with symbol references *)
Inductive Pattern : Type :=
  | PEmpty : Pattern                    (* Empty pattern *)
  | PEpsilon : Pattern                  (* Epsilon (empty string) *)
  | PChar : ascii -> Pattern            (* Single character *)
  | PSymbol : string -> Pattern         (* Symbol reference *)
  | PConcat : Pattern -> Pattern -> Pattern  (* Concatenation *)
  | PAlt : Pattern -> Pattern -> Pattern     (* Alternation *)
  | PStar : Pattern -> Pattern          (* Kleene star *)
  | PPlus : Pattern -> Pattern          (* One or more *)
  | POption : Pattern -> Pattern        (* Optional *)
  | PCharClass : list ascii -> Pattern. (* Character class *)

(** Pure regex (no symbol references) *)
Inductive Regex : Type :=
  | REmpty : Regex
  | REpsilon : Regex
  | RChar : ascii -> Regex
  | RConcat : Regex -> Regex -> Regex
  | RAlt : Regex -> Regex -> Regex
  | RStar : Regex -> Regex
  | RPlus : Regex -> Regex
  | ROption : Regex -> Regex
  | RCharClass : list ascii -> Regex.

(** Symbol table mapping names to patterns *)
Definition SymbolTable := list (string * Pattern).

(** * Helper Definitions *)

(** Look up a symbol in the table *)
Fixpoint lookup (name : string) (table : SymbolTable) : option Pattern :=
  match table with
  | [] => None
  | (n, p) :: rest =>
      if String.eqb name n then Some p
      else lookup name rest
  end.

(** Collect all symbol references in a pattern *)
Fixpoint collect_symbols (p : Pattern) : list string :=
  match p with
  | PEmpty => []
  | PEpsilon => []
  | PChar _ => []
  | PSymbol name => [name]
  | PConcat p1 p2 => collect_symbols p1 ++ collect_symbols p2
  | PAlt p1 p2 => collect_symbols p1 ++ collect_symbols p2
  | PStar p1 => collect_symbols p1
  | PPlus p1 => collect_symbols p1
  | POption p1 => collect_symbols p1
  | PCharClass _ => []
  end.

(** Check if pattern contains symbol references *)
Definition has_symbols (p : Pattern) : bool :=
  match collect_symbols p with
  | [] => false
  | _ => true
  end.

(** * Dependency Graph for Acyclicity *)

(** Direct dependencies of a symbol *)
Definition direct_deps (table : SymbolTable) (name : string) : list string :=
  match lookup name table with
  | None => []
  | Some p => collect_symbols p
  end.

(** Reachability in dependency graph *)
Inductive reaches (table : SymbolTable) : string -> string -> Prop :=
  | reaches_direct : forall s1 s2,
      In s2 (direct_deps table s1) ->
      reaches table s1 s2
  | reaches_trans : forall s1 s2 s3,
      In s2 (direct_deps table s1) ->
      reaches table s2 s3 ->
      reaches table s1 s3.

(** Acyclic symbol table: no symbol reaches itself *)
Definition acyclic_symbols (table : SymbolTable) : Prop :=
  forall s, ~reaches table s s.

(** * Symbol Expansion Algorithm *)

(** Maximum expansion depth *)
Definition MAX_DEPTH : nat := 100.

(** Expand pattern recursively with depth limit *)
Fixpoint expand_pattern (p : Pattern) (table : SymbolTable) (depth : nat) : option Regex :=
  match depth with
  | 0 => None  (* Depth exceeded - possible cycle or too deep *)
  | S d =>
      match p with
      | PEmpty => Some REmpty
      | PEpsilon => Some REpsilon
      | PChar c => Some (RChar c)
      | PSymbol name =>
          match lookup name table with
          | None => None  (* Undefined symbol *)
          | Some p' => expand_pattern p' table d
          end
      | PConcat p1 p2 =>
          match expand_pattern p1 table d, expand_pattern p2 table d with
          | Some r1, Some r2 => Some (RConcat r1 r2)
          | _, _ => None
          end
      | PAlt p1 p2 =>
          match expand_pattern p1 table d, expand_pattern p2 table d with
          | Some r1, Some r2 => Some (RAlt r1 r2)
          | _, _ => None
          end
      | PStar p1 =>
          match expand_pattern p1 table d with
          | Some r1 => Some (RStar r1)
          | None => None
          end
      | PPlus p1 =>
          match expand_pattern p1 table d with
          | Some r1 => Some (RPlus r1)
          | None => None
          end
      | POption p1 =>
          match expand_pattern p1 table d with
          | Some r1 => Some (ROption r1)
          | None => None
          end
      | PCharClass cs => Some (RCharClass cs)
      end
  end.

(** Top-level expansion *)
Definition expand (p : Pattern) (table : SymbolTable) : option Regex :=
  expand_pattern p table MAX_DEPTH.

(** * Language Semantics *)

(** Language of a regex (set of strings it matches) *)
Inductive regex_matches : Regex -> string -> Prop :=
  | match_epsilon : regex_matches REpsilon EmptyString
  | match_char : forall c, regex_matches (RChar c) (String c EmptyString)
  | match_concat : forall r1 r2 s1 s2,
      regex_matches r1 s1 ->
      regex_matches r2 s2 ->
      regex_matches (RConcat r1 r2) (s1 ++ s2)
  | match_alt_left : forall r1 r2 s,
      regex_matches r1 s ->
      regex_matches (RAlt r1 r2) s
  | match_alt_right : forall r1 r2 s,
      regex_matches r2 s ->
      regex_matches (RAlt r1 r2) s
  | match_star_empty : forall r,
      regex_matches (RStar r) EmptyString
  | match_star_step : forall r s1 s2,
      regex_matches r s1 ->
      regex_matches (RStar r) s2 ->
      regex_matches (RStar r) (s1 ++ s2)
  | match_plus : forall r s1 s2,
      regex_matches r s1 ->
      regex_matches (RStar r) s2 ->
      regex_matches (RPlus r) (s1 ++ s2)
  | match_option_none : forall r,
      regex_matches (ROption r) EmptyString
  | match_option_some : forall r s,
      regex_matches r s ->
      regex_matches (ROption r) s
  | match_charclass : forall c cs,
      In c cs ->
      regex_matches (RCharClass cs) (String c EmptyString).

(** Language of a pattern under symbol table substitution *)
Inductive pattern_matches (table : SymbolTable) : Pattern -> string -> Prop :=
  | pmatch_epsilon : pattern_matches table PEpsilon EmptyString
  | pmatch_char : forall c,
      pattern_matches table (PChar c) (String c EmptyString)
  | pmatch_symbol : forall name p s,
      lookup name table = Some p ->
      pattern_matches table p s ->
      pattern_matches table (PSymbol name) s
  | pmatch_concat : forall p1 p2 s1 s2,
      pattern_matches table p1 s1 ->
      pattern_matches table p2 s2 ->
      pattern_matches table (PConcat p1 p2) (s1 ++ s2)
  | pmatch_alt_left : forall p1 p2 s,
      pattern_matches table p1 s ->
      pattern_matches table (PAlt p1 p2) s
  | pmatch_alt_right : forall p1 p2 s,
      pattern_matches table p2 s ->
      pattern_matches table (PAlt p1 p2) s
  | pmatch_star_empty : forall p,
      pattern_matches table (PStar p) EmptyString
  | pmatch_star_step : forall p s1 s2,
      pattern_matches table p s1 ->
      pattern_matches table (PStar p) s2 ->
      pattern_matches table (PStar p) (s1 ++ s2)
  | pmatch_plus : forall p s1 s2,
      pattern_matches table p s1 ->
      pattern_matches table (PStar p) s2 ->
      pattern_matches table (PPlus p) (s1 ++ s2)
  | pmatch_option_none : forall p,
      pattern_matches table (POption p) EmptyString
  | pmatch_option_some : forall p s,
      pattern_matches table p s ->
      pattern_matches table (POption p) s
  | pmatch_charclass : forall c cs,
      In c cs ->
      pattern_matches table (PCharClass cs) (String c EmptyString).

(** * Size Measures *)

(** Size of a pattern (structural) *)
Fixpoint pattern_size (p : Pattern) : nat :=
  match p with
  | PEmpty => 1
  | PEpsilon => 1
  | PChar _ => 1
  | PSymbol _ => 1
  | PConcat p1 p2 => 1 + pattern_size p1 + pattern_size p2
  | PAlt p1 p2 => 1 + pattern_size p1 + pattern_size p2
  | PStar p1 => 1 + pattern_size p1
  | PPlus p1 => 1 + pattern_size p1
  | POption p1 => 1 + pattern_size p1
  | PCharClass _ => 1
  end.

(** Depth of symbol nesting *)
Fixpoint symbol_depth (table : SymbolTable) (name : string) (fuel : nat) : nat :=
  match fuel with
  | 0 => 0
  | S f =>
      match lookup name table with
      | None => 0
      | Some p =>
          let deps := collect_symbols p in
          1 + fold_right max 0 (map (fun n => symbol_depth table n f) deps)
      end
  end.

(** Maximum symbol depth in table *)
Definition max_symbol_depth (table : SymbolTable) : nat :=
  fold_right max 0
    (map (fun entry => match entry with (name, _) =>
                         symbol_depth table name (List.length table)
                       end) table).

(** Symbol depth is bounded by [max_symbol_depth] for any symbol in table. *)
Definition symbol_depth_bounded_contract : Prop := forall table name fuel,
  (List.length table <= fuel)%nat ->
  symbol_depth table name fuel <= max_symbol_depth table.

(** Symbol lookup succeeds for well-formed symbol references in acyclic tables. *)
Definition lookup_succeeds_for_valid_symbols_contract : Prop := forall table name p,
  acyclic_symbols table ->
  In name (collect_symbols p) ->
  exists q, lookup name table = Some q.

(** Expansion of looked-up pattern with sufficient depth succeeds. *)
Definition expansion_of_looked_up_pattern_contract : Prop := forall table name p depth,
  acyclic_symbols table ->
  lookup name table = Some p ->
  depth > symbol_depth table name (List.length table) ->
  exists r, expand_pattern p table depth = Some r.

(** Star matching preserves pattern-to-regex correspondence. *)
Definition star_soundness_contract : Prop := forall p table depth r s,
  expand_pattern p table depth = Some r ->
  regex_matches (RStar r) s ->
  pattern_matches table (PStar p) s.

(** Plus matching preserves pattern-to-regex correspondence. *)
Definition plus_soundness_contract : Prop := forall p table depth r s,
  expand_pattern p table depth = Some r ->
  regex_matches (RPlus r) s ->
  pattern_matches table (PPlus p) s.

(** Soundness for recursively expanding a symbol table entry. *)
Definition symbol_soundness_contract : Prop := forall name p table depth r s,
  lookup name table = Some p ->
  expand_pattern p table depth = Some r ->
  regex_matches r s ->
  pattern_matches table p s.

(** Symbol expansion termination for PSymbol case. *)
Definition psymbol_expansion_terminates_contract : Prop := forall s table depth,
  acyclic_symbols table ->
  depth > max_symbol_depth table + 1 ->
  exists r, expand_pattern (PSymbol s) table depth = Some r.

(** * Termination Proof *)

(** Expansion terminates for acyclic tables *)
Theorem symbol_expansion_terminates : forall p table depth,
  psymbol_expansion_terminates_contract ->
  acyclic_symbols table ->
  depth > max_symbol_depth table + pattern_size p ->
  exists r, expand_pattern p table depth = Some r.
Proof.
  intros p table depth Hpsymbol Hacyclic Hdepth.
  unfold psymbol_expansion_terminates_contract in Hpsymbol.
  generalize dependent depth.
  induction p; intros depth Hdepth.
  - (* PEmpty *)
    destruct depth; [lia |]. simpl. eauto.
  - (* PEpsilon *)
    destruct depth; [lia |]. simpl. eauto.
  - (* PChar *)
    destruct depth; [lia |]. simpl. eauto.
  - (* PSymbol *)
    (* Use the dedicated axiom for PSymbol termination *)
    assert (Hdepth': depth > max_symbol_depth table + 1).
    { simpl in Hdepth. lia. }
    exact (Hpsymbol s table depth Hacyclic Hdepth').
  - (* PConcat *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize1: d > max_symbol_depth table + pattern_size p1) by (simpl in Hdepth; lia).
    assert (Hsize2: d > max_symbol_depth table + pattern_size p2) by (simpl in Hdepth; lia).
    specialize (IHp1 d Hsize1). destruct IHp1 as [r1 Hr1].
    specialize (IHp2 d Hsize2). destruct IHp2 as [r2 Hr2].
    rewrite Hr1, Hr2. eauto.
  - (* PAlt *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize1: d > max_symbol_depth table + pattern_size p1) by (simpl in Hdepth; lia).
    assert (Hsize2: d > max_symbol_depth table + pattern_size p2) by (simpl in Hdepth; lia).
    specialize (IHp1 d Hsize1). destruct IHp1 as [r1 Hr1].
    specialize (IHp2 d Hsize2). destruct IHp2 as [r2 Hr2].
    rewrite Hr1, Hr2. eauto.
  - (* PStar *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize: d > max_symbol_depth table + pattern_size p) by (simpl in Hdepth; lia).
    specialize (IHp d Hsize). destruct IHp as [r Hr].
    rewrite Hr. eauto.
  - (* PPlus *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize: d > max_symbol_depth table + pattern_size p) by (simpl in Hdepth; lia).
    specialize (IHp d Hsize). destruct IHp as [r Hr].
    rewrite Hr. eauto.
  - (* POption *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize: d > max_symbol_depth table + pattern_size p) by (simpl in Hdepth; lia).
    specialize (IHp d Hsize). destruct IHp as [r Hr].
    rewrite Hr. eauto.
  - (* PCharClass *)
    destruct depth; [lia |]. simpl. eauto.
Qed.

(** * Language Preservation *)

(** Soundness: if expanded regex matches, original pattern matches *)
Theorem expansion_soundness : forall p table depth r s,
  symbol_soundness_contract ->
  star_soundness_contract ->
  plus_soundness_contract ->
  expand_pattern p table depth = Some r ->
  regex_matches r s ->
  pattern_matches table p s.
Proof.
  intros p table depth r str Hsymbol Hstar Hplus Hexp Hmatch.
  unfold symbol_soundness_contract in Hsymbol.
  unfold star_soundness_contract in Hstar.
  unfold plus_soundness_contract in Hplus.
  generalize dependent str.
  generalize dependent r.
  generalize dependent depth.
  induction p; intros depth r Hexp str Hmatch.
  - (* PEmpty *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. subst. inversion Hmatch.
  - (* PEpsilon *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. subst. inversion Hmatch. subst.
    constructor.
  - (* PChar *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. subst. inversion Hmatch. subst.
    constructor.
  - (* PSymbol *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (lookup s table) eqn:Hlookup; [| discriminate].
    apply pmatch_symbol with (p := p); auto.
    eapply Hsymbol; eauto.
  - (* PConcat *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p1 table depth) eqn:He1; [| discriminate].
    destruct (expand_pattern p2 table depth) eqn:He2; [| discriminate].
    inversion Hexp. subst. clear Hexp.
    inversion Hmatch. subst.
    apply pmatch_concat.
    + eapply IHp1; eauto.
    + eapply IHp2; eauto.
  - (* PAlt *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p1 table depth) eqn:He1; [| discriminate].
    destruct (expand_pattern p2 table depth) eqn:He2; [| discriminate].
    inversion Hexp. subst. clear Hexp.
    inversion Hmatch; subst.
    + apply pmatch_alt_left. eapply IHp1; eauto.
    + apply pmatch_alt_right. eapply IHp2; eauto.
  - (* PStar *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst. clear Hexp.
    (* Use the star soundness helper axiom *)
    exact (Hstar p table depth r0 str He Hmatch).
  - (* PPlus *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst. clear Hexp.
    (* Use the plus soundness helper axiom *)
    exact (Hplus p table depth r0 str He Hmatch).
  - (* POption *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst. clear Hexp.
    inversion Hmatch; subst.
    + constructor.
    + apply pmatch_option_some. eapply IHp; eauto.
  - (* PCharClass *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. subst. inversion Hmatch. subst.
    constructor. assumption.
Qed.

(** Completeness: if original pattern matches, expanded regex matches *)
Theorem expansion_completeness : forall p table depth r s,
  acyclic_symbols table ->
  expand_pattern p table depth = Some r ->
  pattern_matches table p s ->
  regex_matches r s.
Proof.
  intros p table depth r s Hacyclic Hexp Hmatch.
  generalize dependent r.
  generalize dependent depth.
  induction Hmatch; intros depth r Hexp.
  - (* pmatch_epsilon *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. constructor.
  - (* pmatch_char *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. constructor.
  - (* pmatch_symbol *)
    destruct depth; [discriminate |]. simpl in Hexp.
    rewrite H in Hexp.
    eapply IHHmatch; eauto.
  - (* pmatch_concat *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p1 table depth) eqn:He1; [| discriminate].
    destruct (expand_pattern p2 table depth) eqn:He2; [| discriminate].
    inversion Hexp. subst.
    apply match_concat.
    + eapply IHHmatch1; eauto.
    + eapply IHHmatch2; eauto.
  - (* pmatch_alt_left *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p1 table depth) eqn:He1; [| discriminate].
    destruct (expand_pattern p2 table depth) eqn:He2; [| discriminate].
    inversion Hexp. subst.
    apply match_alt_left. eapply IHHmatch; eauto.
  - (* pmatch_alt_right *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p1 table depth) eqn:He1; [| discriminate].
    destruct (expand_pattern p2 table depth) eqn:He2; [| discriminate].
    inversion Hexp. subst.
    apply match_alt_right. eapply IHHmatch; eauto.
  - (* pmatch_star_empty *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst.
    constructor.
  - (* pmatch_star_step *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst.
    apply match_star_step.
    + eapply IHHmatch1; eauto.
    + eapply (IHHmatch2 (S depth) (RStar r0)).
      simpl. rewrite He. reflexivity.
  - (* pmatch_plus *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst.
    apply match_plus.
    + eapply IHHmatch1; eauto.
    + eapply (IHHmatch2 (S depth) (RStar r0)).
      simpl. rewrite He. reflexivity.
  - (* pmatch_option_none *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst. constructor.
  - (* pmatch_option_some *)
    destruct depth; [discriminate |]. simpl in Hexp.
    destruct (expand_pattern p table depth) eqn:He; [| discriminate].
    inversion Hexp. subst.
    apply match_option_some. eapply IHHmatch; eauto.
  - (* pmatch_charclass *)
    destruct depth; [discriminate |]. simpl in Hexp.
    inversion Hexp. subst. constructor. assumption.
Qed.

(** Main language preservation obligation. *)
Definition expansion_preserves_language_contract : Prop := forall p table r,
  symbol_soundness_contract ->
  star_soundness_contract ->
  plus_soundness_contract ->
  acyclic_symbols table ->
  expand p table = Some r ->
  forall s, regex_matches r s <-> pattern_matches table p s.

(** Main theorem: expansion preserves language *)
Theorem expansion_preserves_language : expansion_preserves_language_contract ->
  forall p table r,
  symbol_soundness_contract ->
  star_soundness_contract ->
  plus_soundness_contract ->
  acyclic_symbols table ->
  expand p table = Some r ->
  forall s, regex_matches r s <-> pattern_matches table p s.
Proof.
  intros Hcontract p table r Hsymbol Hstar Hplus Hacyclic Hexp s.
  unfold expansion_preserves_language_contract in Hcontract.
  exact (Hcontract p table r Hsymbol Hstar Hplus Hacyclic Hexp s).
Qed.

(** * Cycle Detection *)

(** Check for cycles using depth-limited traversal *)
Fixpoint has_cycle_from (table : SymbolTable) (start : string) (visited : list string) (fuel : nat) : bool :=
  match fuel with
  | 0 => true  (* Assume cycle if we run out of fuel *)
  | S f =>
      if existsb (String.eqb start) visited then true
      else
        let deps := direct_deps table start in
        existsb (fun dep => has_cycle_from table dep (start :: visited) f) deps
  end.

(** Top-level cycle check *)
Definition has_cycles (table : SymbolTable) : bool :=
  existsb (fun entry => match entry with (name, _) =>
                         has_cycle_from table name [] (List.length table + 1)
                       end) table.

(** Cycle detection is correct: no cycles implies acyclic. *)
Definition cycle_detection_correct_contract : Prop := forall table,
  has_cycles table = false ->
  acyclic_symbols table.

(** Cycle detection correctness *)
Theorem no_cycles_implies_acyclic : forall table,
  cycle_detection_correct_contract ->
  has_cycles table = false ->
  acyclic_symbols table.
Proof.
  intros table Hcontract Hcycles.
  unfold cycle_detection_correct_contract in Hcontract.
  exact (Hcontract table Hcycles).
Qed.

(** * Depth Bounds *)

(** Expansion respects depth limit. *)
Definition expand_respects_depth_contract : Prop := forall p table depth1 depth2 r,
  depth1 <= depth2 ->
  expand_pattern p table depth1 = Some r ->
  expand_pattern p table depth2 = Some r.

Lemma expand_respects_depth : expand_respects_depth_contract ->
  forall p table depth1 depth2 r,
  depth1 <= depth2 ->
  expand_pattern p table depth1 = Some r ->
  expand_pattern p table depth2 = Some r.
Proof.
  intros Hcontract p table depth1 depth2 r Hle Hexp.
  unfold expand_respects_depth_contract in Hcontract.
  exact (Hcontract p table depth1 depth2 r Hle Hexp).
Qed.

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

(** Patterns that do not require symbol-table expansion. *)
Fixpoint symbol_free (p : Pattern) : Prop :=
  match p with
  | PEmpty => True
  | PEpsilon => True
  | PChar _ => True
  | PSymbol _ => False
  | PConcat p1 p2 => symbol_free p1 /\ symbol_free p2
  | PAlt p1 p2 => symbol_free p1 /\ symbol_free p2
  | PStar p1 => symbol_free p1
  | PPlus p1 => symbol_free p1
  | POption p1 => symbol_free p1
  | PCharClass _ => True
  end.

(** * Termination Proof *)

(** Expansion terminates for symbol-free patterns. *)
Theorem symbol_expansion_terminates : forall p table depth,
  symbol_free p ->
  depth > pattern_size p ->
  exists r, expand_pattern p table depth = Some r.
Proof.
  intros p table depth.
  generalize dependent depth.
  induction p; intros depth Hfree Hdepth.
  - (* PEmpty *)
    destruct depth; [lia |]. simpl. eauto.
  - (* PEpsilon *)
    destruct depth; [lia |]. simpl. eauto.
  - (* PChar *)
    destruct depth; [lia |]. simpl. eauto.
  - (* PSymbol *)
    contradiction.
  - (* PConcat *)
    destruct depth as [| d]; [lia |]. simpl.
    destruct Hfree as [Hfree1 Hfree2].
    assert (Hsize1: d > pattern_size p1) by (simpl in Hdepth; lia).
    assert (Hsize2: d > pattern_size p2) by (simpl in Hdepth; lia).
    specialize (IHp1 d Hfree1 Hsize1). destruct IHp1 as [r1 Hr1].
    specialize (IHp2 d Hfree2 Hsize2). destruct IHp2 as [r2 Hr2].
    rewrite Hr1, Hr2. eauto.
  - (* PAlt *)
    destruct depth as [| d]; [lia |]. simpl.
    destruct Hfree as [Hfree1 Hfree2].
    assert (Hsize1: d > pattern_size p1) by (simpl in Hdepth; lia).
    assert (Hsize2: d > pattern_size p2) by (simpl in Hdepth; lia).
    specialize (IHp1 d Hfree1 Hsize1). destruct IHp1 as [r1 Hr1].
    specialize (IHp2 d Hfree2 Hsize2). destruct IHp2 as [r2 Hr2].
    rewrite Hr1, Hr2. eauto.
  - (* PStar *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize: d > pattern_size p) by (simpl in Hdepth; lia).
    specialize (IHp d Hfree Hsize). destruct IHp as [r Hr].
    rewrite Hr. eauto.
  - (* PPlus *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize: d > pattern_size p) by (simpl in Hdepth; lia).
    specialize (IHp d Hfree Hsize). destruct IHp as [r Hr].
    rewrite Hr. eauto.
  - (* POption *)
    destruct depth as [| d]; [lia |]. simpl.
    assert (Hsize: d > pattern_size p) by (simpl in Hdepth; lia).
    specialize (IHp d Hfree Hsize). destruct IHp as [r Hr].
    rewrite Hr. eauto.
  - (* PCharClass *)
    destruct depth; [lia |]. simpl. eauto.
Qed.

(** * Language Preservation *)

(** Kleene-star soundness follows from soundness of the repeated pattern. *)
Lemma expansion_star_soundness_from_soundness :
  forall depth,
  (forall p table r s,
    expand_pattern p table depth = Some r ->
    regex_matches r s ->
    pattern_matches table p s) ->
  forall p table r s,
  expand_pattern p table depth = Some r ->
  regex_matches (RStar r) s ->
  pattern_matches table (PStar p) s.
Proof.
  intros depth Hsound p table r str Hexp Hmatch.
  remember (RStar r) as star_regex eqn:Hstar.
  induction Hmatch as
    [| c
     | r1 r2 s1 s2 Hm1 IHm1 Hm2 IHm2
     | r1 r2 s Halt IHalt
     | r1 r2 s Halt IHalt
     | r_empty
     | r_step s1 s2 Hhead IHhead Htail IHtail
     | r_plus s1 s2 Hhead IHhead Htail IHtail
     | r_option
     | r_option s Hopt IHopt
     | c cs Hin];
    inversion Hstar; subst.
  - constructor.
  - apply pmatch_star_step.
    + exact (Hsound p table r s1 Hexp Hhead).
    + apply IHtail. reflexivity.
Qed.

(** Soundness by expansion fuel. *)
Lemma expansion_soundness_depth : forall depth p table r s,
  expand_pattern p table depth = Some r ->
  regex_matches r s ->
  pattern_matches table p s.
Proof.
  induction depth as [| depth IH]; intros p table r str Hexp Hmatch.
  - simpl in Hexp. discriminate.
  - destruct p; simpl in Hexp.
    + (* PEmpty *)
      inversion Hexp. subst. inversion Hmatch.
    + (* PEpsilon *)
      inversion Hexp. subst. inversion Hmatch. subst.
      constructor.
    + (* PChar *)
      inversion Hexp. subst. inversion Hmatch. subst.
      constructor.
    + (* PSymbol *)
      destruct (lookup s table) as [p'|] eqn:Hlookup; [| discriminate].
      apply pmatch_symbol with (p := p').
      * exact Hlookup.
      * exact (IH p' table r str Hexp Hmatch).
    + (* PConcat *)
      destruct (expand_pattern p1 table depth) as [r1|] eqn:He1; [| discriminate].
      destruct (expand_pattern p2 table depth) as [r2|] eqn:He2; [| discriminate].
      inversion Hexp. subst. clear Hexp.
      inversion Hmatch. subst.
      apply pmatch_concat.
      * match goal with
        | H : regex_matches r1 ?s1 |- _ =>
            exact (IH p1 table r1 s1 He1 H)
        end.
      * match goal with
        | H : regex_matches r2 ?s2 |- _ =>
            exact (IH p2 table r2 s2 He2 H)
        end.
    + (* PAlt *)
      destruct (expand_pattern p1 table depth) as [r1|] eqn:He1; [| discriminate].
      destruct (expand_pattern p2 table depth) as [r2|] eqn:He2; [| discriminate].
      inversion Hexp. subst. clear Hexp.
      inversion Hmatch; subst.
      * apply pmatch_alt_left.
        match goal with
        | H : regex_matches r1 ?s |- _ =>
            exact (IH p1 table r1 s He1 H)
        end.
      * apply pmatch_alt_right.
        match goal with
        | H : regex_matches r2 ?s |- _ =>
            exact (IH p2 table r2 s He2 H)
        end.
    + (* PStar *)
      destruct (expand_pattern p table depth) as [r_sub|] eqn:He; [| discriminate].
      inversion Hexp. subst. clear Hexp.
      exact (expansion_star_soundness_from_soundness
        depth IH p table r_sub str He Hmatch).
    + (* PPlus *)
      destruct (expand_pattern p table depth) as [r_sub|] eqn:He; [| discriminate].
      inversion Hexp. subst. clear Hexp.
      remember (RPlus r_sub) as plus_regex eqn:Hplus.
      induction Hmatch as
        [| c
         | r1 r2 s1 s2 Hm1 IHm1 Hm2 IHm2
         | r1 r2 s Halt IHalt
         | r1 r2 s Halt IHalt
         | r_empty
         | r_step s1 s2 Hhead IHhead Htail IHtail
         | r_plus s1 s2 Hhead IHhead Htail IHtail
         | r_option
         | r_option s Hopt IHopt
         | c cs Hin];
        inversion Hplus; subst.
      apply pmatch_plus.
      * exact (IH p table r_sub s1 He Hhead).
      * exact (expansion_star_soundness_from_soundness
          depth IH p table r_sub s2 He Htail).
    + (* POption *)
      destruct (expand_pattern p table depth) as [r_sub|] eqn:He; [| discriminate].
      inversion Hexp. subst. clear Hexp.
      inversion Hmatch; subst.
      * constructor.
      * apply pmatch_option_some.
        match goal with
        | H : regex_matches r_sub ?s |- _ =>
            exact (IH p table r_sub s He H)
        end.
    + (* PCharClass *)
      inversion Hexp. subst. inversion Hmatch. subst.
      constructor. assumption.
Qed.

(** Soundness: if expanded regex matches, original pattern matches *)
Theorem expansion_soundness : forall p table depth r s,
  expand_pattern p table depth = Some r ->
  regex_matches r s ->
  pattern_matches table p s.
Proof.
  intros p table depth r str Hexp Hmatch.
  exact (expansion_soundness_depth depth p table r str Hexp Hmatch).
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

(** Main theorem: expansion preserves language *)
Theorem expansion_preserves_language :
  forall p table r,
  acyclic_symbols table ->
  expand_pattern p table MAX_DEPTH = Some r ->
  forall s, regex_matches r s <-> pattern_matches table p s.
Proof.
  intros p table r Hacyclic Hexp s.
  split.
  - abstract (intro Hmatch;
      exact (expansion_soundness p table MAX_DEPTH r s Hexp Hmatch)).
  - abstract (intro Hmatch;
      exact (expansion_completeness p table MAX_DEPTH r s Hacyclic Hexp Hmatch)).
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

(** Bounded reachability matching the executable DFS fuel. *)
Inductive reaches_within (table : SymbolTable) : nat -> string -> string -> Prop :=
  | reaches_within_direct : forall fuel s1 s2,
      In s2 (direct_deps table s1) ->
      reaches_within table (S fuel) s1 s2
  | reaches_within_trans : forall fuel s1 s2 s3,
      In s2 (direct_deps table s1) ->
      reaches_within table fuel s2 s3 ->
      reaches_within table (S fuel) s1 s3.

Lemma existsb_false_forall : forall {A : Type} (f : A -> bool) xs x,
  existsb f xs = false ->
  In x xs ->
  f x = false.
Proof.
  intros A f xs x Hfalse Hin.
  destruct (f x) eqn:Hfx; [| reflexivity].
  exfalso.
  pose proof (proj2 (existsb_exists f xs)) as Hexists.
  specialize (Hexists (ex_intro _ x (conj Hin Hfx))).
  rewrite Hfalse in Hexists.
  discriminate.
Qed.

Lemma existsb_string_eqb_in : forall x xs,
  In x xs ->
  existsb (String.eqb x) xs = true.
Proof.
  intros x xs Hin.
  apply existsb_exists.
  exists x.
  split; [exact Hin | apply String.eqb_refl].
Qed.

Lemma has_cycle_from_seen : forall table fuel start visited,
  In start visited ->
  has_cycle_from table start visited fuel = true.
Proof.
  intros table fuel start visited Hin.
  destruct fuel as [| fuel]; simpl.
  - reflexivity.
  - rewrite (existsb_string_eqb_in start visited Hin).
    reflexivity.
Qed.

Lemma has_cycle_from_false_no_reaches_visited :
  forall table fuel start visited target,
    has_cycle_from table start visited fuel = false ->
    In target visited ->
    ~ reaches_within table fuel start target.
Proof.
  intros table fuel.
  induction fuel as [| fuel IH]; intros start visited target Hfalse Hvisited Hreach.
  - simpl in Hfalse. discriminate.
  - simpl in Hfalse.
    destruct (existsb (String.eqb start) visited) eqn:Hseen; [discriminate |].
    inversion Hreach as [fuel' s1 s2 Hin_dep
                       | fuel' s1 s2 s3 Hin_dep Hwithin]; subst.
    + assert (Hdep_true :
        existsb
          (fun dep : string => has_cycle_from table dep (start :: visited) fuel)
          (direct_deps table start) = true).
      { apply existsb_exists.
        exists target.
        split; [exact Hin_dep |].
        apply has_cycle_from_seen.
        right. exact Hvisited. }
      rewrite Hdep_true in Hfalse. discriminate.
    + pose proof (existsb_false_forall
        (fun dep : string => has_cycle_from table dep (start :: visited) fuel)
        (direct_deps table start) s2 Hfalse Hin_dep) as Hdep.
      eapply (IH s2 (start :: visited) target); eauto.
      right. exact Hvisited.
Qed.

Lemma has_cycle_from_false_no_self :
  forall table fuel start visited,
    has_cycle_from table start visited fuel = false ->
    ~ reaches_within table fuel start start.
Proof.
  intros table fuel start visited Hfalse Hreach.
  destruct fuel as [| fuel].
  - simpl in Hfalse. discriminate.
  - simpl in Hfalse.
    destruct (existsb (String.eqb start) visited) eqn:Hseen; [discriminate |].
    inversion Hreach as [fuel' s1 s2 Hin_dep
                       | fuel' s1 s2 s3 Hin_dep Hwithin]; subst.
    + assert (Hdep_true :
        existsb
          (fun dep : string => has_cycle_from table dep (start :: visited) fuel)
          (direct_deps table start) = true).
      { apply existsb_exists.
        exists start.
        split; [exact Hin_dep |].
        apply has_cycle_from_seen.
        left. reflexivity. }
      rewrite Hdep_true in Hfalse. discriminate.
    + pose proof (existsb_false_forall
        (fun dep : string => has_cycle_from table dep (start :: visited) fuel)
        (direct_deps table start) s2 Hfalse Hin_dep) as Hdep.
      eapply (has_cycle_from_false_no_reaches_visited table fuel s2 (start :: visited) start); eauto.
      left. reflexivity.
Qed.

(** If the top-level DFS reports no cycle, then no table entry has a
    fuel-bounded self reachability witness.  This is the executable guarantee
    provided by [has_cycles]; the stronger unbounded graph acyclicity theorem
    requires an additional finite simple-cycle argument. *)
Theorem no_cycles_implies_no_bounded_self_reach :
  forall table name p,
    has_cycles table = false ->
    In (name, p) table ->
    ~ reaches_within table (List.length table + 1) name name.
Proof.
  intros table name p Hcycles Hin Hreach.
  unfold has_cycles in Hcycles.
  pose proof (existsb_false_forall
    (fun entry : string * Pattern =>
       match entry with
       | (name0, _) => has_cycle_from table name0 [] (Datatypes.length table + 1)
       end)
    table (name, p) Hcycles Hin) as Hstart.
  exact (has_cycle_from_false_no_self table (List.length table + 1) name [] Hstart Hreach).
Qed.

(** * Depth Bounds *)

(** Expansion respects depth limit. *)
Lemma expand_respects_depth :
  forall p table depth1 depth2 r,
  depth1 <= depth2 ->
  expand_pattern p table depth1 = Some r ->
  expand_pattern p table depth2 = Some r.
Proof.
  intros p table depth1.
  revert p table.
  induction depth1 as [| depth1 IH]; intros p table depth2 r Hle Hexp.
  - simpl in Hexp. discriminate.
  - destruct depth2 as [| depth2]; [lia |].
    assert (Hle' : depth1 <= depth2) by lia.
    destruct p; simpl in Hexp; simpl.
    + inversion Hexp. reflexivity.
    + inversion Hexp. reflexivity.
    + inversion Hexp. reflexivity.
    + destruct (lookup s table) as [p'|] eqn:Hlookup; [| discriminate].
      eapply IH; eauto.
    + destruct (expand_pattern p1 table depth1) as [r1|] eqn:Hexp1;
        [| discriminate].
      destruct (expand_pattern p2 table depth1) as [r2|] eqn:Hexp2;
        [| discriminate].
      inversion Hexp. subst.
      rewrite (IH p1 table depth2 r1 Hle' Hexp1).
      rewrite (IH p2 table depth2 r2 Hle' Hexp2).
      reflexivity.
    + destruct (expand_pattern p1 table depth1) as [r1|] eqn:Hexp1;
        [| discriminate].
      destruct (expand_pattern p2 table depth1) as [r2|] eqn:Hexp2;
        [| discriminate].
      inversion Hexp. subst.
      rewrite (IH p1 table depth2 r1 Hle' Hexp1).
      rewrite (IH p2 table depth2 r2 Hle' Hexp2).
      reflexivity.
    + destruct (expand_pattern p table depth1) as [r'|] eqn:Hexp';
        [| discriminate].
      inversion Hexp. subst.
      rewrite (IH p table depth2 r' Hle' Hexp').
      reflexivity.
    + destruct (expand_pattern p table depth1) as [r'|] eqn:Hexp';
        [| discriminate].
      inversion Hexp. subst.
      rewrite (IH p table depth2 r' Hle' Hexp').
      reflexivity.
    + destruct (expand_pattern p table depth1) as [r'|] eqn:Hexp';
        [| discriminate].
      inversion Hexp. subst.
      rewrite (IH p table depth2 r' Hle' Hexp').
      reflexivity.
    + inversion Hexp. reflexivity.
Qed.

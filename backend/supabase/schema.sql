-- biopapers accounts: run once in the Supabase SQL editor (Project → SQL → New query).
--
-- Everything a person does on the site lives here, one row per user, readable and writable only
-- by that user (row-level security). Claude's per-user scores are written by the daily GitHub
-- Action with the service key, which bypasses RLS; users can only read their own.
-- Sign-ups are off (Authentication → Providers → disable "Allow new users to sign up"): you invite
-- people from Authentication → Users.

-- Who someone is and what they want to see.
create table if not exists public.profiles (
  user_id      uuid primary key references auth.users on delete cascade,
  display_name text,
  interests    text not null default '',      -- like config/profile.md: what Claude scores for
  hidden_tabs  text[] not null default '{}',  -- topic tabs they turned off
  onboarded    boolean not null default false,
  updated_at   timestamptz not null default now()
);

-- The site's synced state blob (saves, favorites, hides, clicks, explored, settings…), merged on the
-- client per entry (mergeSynced in site/app.js). `rev` makes writes safe across devices: a client
-- writes with `where rev = <the rev it read>` and, if no row matched, re-reads, merges and retries.
create table if not exists public.user_state (
  user_id    uuid primary key references auth.users on delete cascade,
  state      jsonb not null default '{}',
  rev        bigint not null default 0,
  updated_at timestamptz not null default now()
);

-- On-demand take-homes: kept out of the state blob because each is several KB.
create table if not exists public.summaries (
  user_id  uuid not null references auth.users on delete cascade,
  paper_id text not null,
  data     jsonb not null,
  ts       timestamptz not null default now(),
  primary key (user_id, paper_id)
);

-- Claude's relevance score of a paper for this user. `v` is the version of the user's interests
-- it was scored against (a hash); a changed profile re-scores. Written only by the daily Action.
create table if not exists public.scores (
  user_id   uuid not null references auth.users on delete cascade,
  paper_id  text not null,
  score     smallint,          -- null = the safety filter declined to score it
  why       text,
  v         text not null,
  scored_at timestamptz not null default now(),
  primary key (user_id, paper_id)
);
create index if not exists scores_scored_at on public.scores (scored_at);

-- Row-level security: each person sees only their own rows.
alter table public.profiles   enable row level security;
alter table public.user_state enable row level security;
alter table public.summaries  enable row level security;
alter table public.scores     enable row level security;

drop policy if exists own on public.profiles;
create policy own on public.profiles for all
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
drop policy if exists own on public.user_state;
create policy own on public.user_state for all
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
drop policy if exists own on public.summaries;
create policy own on public.summaries for all
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
drop policy if exists own_read on public.scores;
create policy own_read on public.scores for select
  using ((select auth.uid()) = user_id);   -- no insert/update policy: only the service key writes

-- Every invited user gets an empty profile and state row.
create or replace function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  insert into public.profiles (user_id, display_name)
    values (new.id, coalesce(new.raw_user_meta_data->>'name', split_part(new.email, '@', 1)))
    on conflict do nothing;
  insert into public.user_state (user_id) values (new.id) on conflict do nothing;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

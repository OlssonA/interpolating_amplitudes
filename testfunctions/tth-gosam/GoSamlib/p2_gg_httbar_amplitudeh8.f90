module    p2_gg_httbar_amplitudeh8
   use p2_gg_httbar_config, only: ki, &
       & reduction_interoperation
   use p2_gg_httbar_color, only: numcs
   use p2_gg_httbar_groups
   use ninjago_module, only: ki_nin
   use p2_gg_httbar_ninjah8
   
   implicit none
   private

   public :: finite_renormalisation, samplitude
contains
!---#[ function finite_renormalisation:
   function     finite_renormalisation(scale2) result(amp)
      use p2_gg_httbar_util, only: square
      use p2_gg_httbar_color, only: CF, CA
      use p2_gg_httbar_kinematics, only: &
      & num_light_quarks, num_gluons
      use p2_gg_httbar_diagramsh8l0, only: amplitudel0 => amplitude
      implicit none
      real(ki), intent(in) :: scale2
      real(ki) :: amp
      amp = 0.0_ki
   end function finite_renormalisation
   !---#] function finite_renormalisation:

   !---#[ function samplitude:
   function     samplitude(scale2,ok,rational2,opt_amp0,opt_perm)
      use p2_gg_httbar_config, only: include_eps_terms, include_eps2_terms, &
      & logfile, debug_nlo_diagrams
      use p2_gg_httbar_globalsl1, only: amp0,perm, use_perm, epspow
      use p2_gg_httbar_globalsh8, &
     & only: init_lo, rat2
      use p2_gg_httbar_abbrevd85h8, only: init_abbrevd85 => init_abbrev
      use p2_gg_httbar_abbrevd262h8, only: init_abbrevd262 => init_abbrev
      use p2_gg_httbar_abbrevd91h8, only: init_abbrevd91 => init_abbrev
      use p2_gg_httbar_abbrevd260h8, only: init_abbrevd260 => init_abbrev
      use p2_gg_httbar_abbrevd29h8, only: init_abbrevd29 => init_abbrev
      use p2_gg_httbar_abbrevd31h8, only: init_abbrevd31 => init_abbrev
      use p2_gg_httbar_abbrevd33h8, only: init_abbrevd33 => init_abbrev
      use p2_gg_httbar_abbrevd74h8, only: init_abbrevd74 => init_abbrev
      use p2_gg_httbar_abbrevd77h8, only: init_abbrevd77 => init_abbrev
      use p2_gg_httbar_abbrevd82h8, only: init_abbrevd82 => init_abbrev
      use p2_gg_httbar_abbrevd90h8, only: init_abbrevd90 => init_abbrev
      use p2_gg_httbar_abbrevd148h8, only: init_abbrevd148 => init_abbrev
      use p2_gg_httbar_abbrevd163h8, only: init_abbrevd163 => init_abbrev
      use p2_gg_httbar_abbrevd258h8, only: init_abbrevd258 => init_abbrev
      use p2_gg_httbar_abbrevd1h8, only: init_abbrevd1 => init_abbrev
      use p2_gg_httbar_abbrevd2h8, only: init_abbrevd2 => init_abbrev
      use p2_gg_httbar_abbrevd3h8, only: init_abbrevd3 => init_abbrev
      use p2_gg_httbar_abbrevd5h8, only: init_abbrevd5 => init_abbrev
      use p2_gg_httbar_abbrevd6h8, only: init_abbrevd6 => init_abbrev
      use p2_gg_httbar_abbrevd7h8, only: init_abbrevd7 => init_abbrev
      use p2_gg_httbar_abbrevd11h8, only: init_abbrevd11 => init_abbrev
      use p2_gg_httbar_abbrevd13h8, only: init_abbrevd13 => init_abbrev
      use p2_gg_httbar_abbrevd26h8, only: init_abbrevd26 => init_abbrev
      use p2_gg_httbar_abbrevd28h8, only: init_abbrevd28 => init_abbrev
      use p2_gg_httbar_abbrevd35h8, only: init_abbrevd35 => init_abbrev
      use p2_gg_httbar_abbrevd46h8, only: init_abbrevd46 => init_abbrev
      use p2_gg_httbar_abbrevd66h8, only: init_abbrevd66 => init_abbrev
      use p2_gg_httbar_abbrevd71h8, only: init_abbrevd71 => init_abbrev
      use p2_gg_httbar_abbrevd80h8, only: init_abbrevd80 => init_abbrev
      use p2_gg_httbar_abbrevd84h8, only: init_abbrevd84 => init_abbrev
      use p2_gg_httbar_abbrevd88h8, only: init_abbrevd88 => init_abbrev
      use p2_gg_httbar_abbrevd133h8, only: init_abbrevd133 => init_abbrev
      use p2_gg_httbar_abbrevd178h8, only: init_abbrevd178 => init_abbrev
      use p2_gg_httbar_abbrevd195h8, only: init_abbrevd195 => init_abbrev
      use p2_gg_httbar_abbrevd256h8, only: init_abbrevd256 => init_abbrev
      use p2_gg_httbar_abbrevd10h8, only: init_abbrevd10 => init_abbrev
      use p2_gg_httbar_abbrevd34h8, only: init_abbrevd34 => init_abbrev
      use p2_gg_httbar_abbrevd36h8, only: init_abbrevd36 => init_abbrev
      use p2_gg_httbar_abbrevd38h8, only: init_abbrevd38 => init_abbrev
      use p2_gg_httbar_abbrevd67h8, only: init_abbrevd67 => init_abbrev
      use p2_gg_httbar_abbrevd79h8, only: init_abbrevd79 => init_abbrev
      use p2_gg_httbar_abbrevd83h8, only: init_abbrevd83 => init_abbrev
      use p2_gg_httbar_abbrevd130h8, only: init_abbrevd130 => init_abbrev
      use p2_gg_httbar_abbrevd132h8, only: init_abbrevd132 => init_abbrev
      use p2_gg_httbar_abbrevd254h8, only: init_abbrevd254 => init_abbrev
      use p2_gg_httbar_abbrevd69h8, only: init_abbrevd69 => init_abbrev
      use p2_gg_httbar_abbrevd78h8, only: init_abbrevd78 => init_abbrev
      use p2_gg_httbar_abbrevd125h8, only: init_abbrevd125 => init_abbrev
      use p2_gg_httbar_abbrevd259h8, only: init_abbrevd259 => init_abbrev
      use p2_gg_httbar_abbrevd30h8, only: init_abbrevd30 => init_abbrev
      use p2_gg_httbar_abbrevd42h8, only: init_abbrevd42 => init_abbrev
      use p2_gg_httbar_abbrevd73h8, only: init_abbrevd73 => init_abbrev
      use p2_gg_httbar_abbrevd81h8, only: init_abbrevd81 => init_abbrev
      use p2_gg_httbar_abbrevd162h8, only: init_abbrevd162 => init_abbrev
      use p2_gg_httbar_abbrevd257h8, only: init_abbrevd257 => init_abbrev
      use p2_gg_httbar_abbrevd12h8, only: init_abbrevd12 => init_abbrev
      use p2_gg_httbar_abbrevd22h8, only: init_abbrevd22 => init_abbrev
      use p2_gg_httbar_abbrevd32h8, only: init_abbrevd32 => init_abbrev
      use p2_gg_httbar_abbrevd37h8, only: init_abbrevd37 => init_abbrev
      use p2_gg_httbar_abbrevd40h8, only: init_abbrevd40 => init_abbrev
      use p2_gg_httbar_abbrevd50h8, only: init_abbrevd50 => init_abbrev
      use p2_gg_httbar_abbrevd68h8, only: init_abbrevd68 => init_abbrev
      use p2_gg_httbar_abbrevd76h8, only: init_abbrevd76 => init_abbrev
      use p2_gg_httbar_abbrevd89h8, only: init_abbrevd89 => init_abbrev
      use p2_gg_httbar_abbrevd101h8, only: init_abbrevd101 => init_abbrev
      use p2_gg_httbar_abbrevd129h8, only: init_abbrevd129 => init_abbrev
      use p2_gg_httbar_abbrevd147h8, only: init_abbrevd147 => init_abbrev
      use p2_gg_httbar_abbrevd172h8, only: init_abbrevd172 => init_abbrev
      use p2_gg_httbar_abbrevd203h8, only: init_abbrevd203 => init_abbrev
      use p2_gg_httbar_abbrevd253h8, only: init_abbrevd253 => init_abbrev
      use p2_gg_httbar_abbrevd27h8, only: init_abbrevd27 => init_abbrev
      use p2_gg_httbar_abbrevd44h8, only: init_abbrevd44 => init_abbrev
      use p2_gg_httbar_abbrevd70h8, only: init_abbrevd70 => init_abbrev
      use p2_gg_httbar_abbrevd87h8, only: init_abbrevd87 => init_abbrev
      use p2_gg_httbar_abbrevd113h8, only: init_abbrevd113 => init_abbrev
      use p2_gg_httbar_abbrevd142h8, only: init_abbrevd142 => init_abbrev
      use p2_gg_httbar_abbrevd177h8, only: init_abbrevd177 => init_abbrev
      use p2_gg_httbar_abbrevd255h8, only: init_abbrevd255 => init_abbrev
      use p2_gg_httbar_abbrevd72h8, only: init_abbrevd72 => init_abbrev
      use p2_gg_httbar_abbrevd75h8, only: init_abbrevd75 => init_abbrev
      use p2_gg_httbar_abbrevd261h8, only: init_abbrevd261 => init_abbrev
      use p2_gg_httbar_abbrevd86h8, only: init_abbrevd86 => init_abbrev
      use p2_gg_httbar_abbrevd264h8, only: init_abbrevd264 => init_abbrev
      use p2_gg_httbar_abbrevd92h8, only: init_abbrevd92 => init_abbrev
      use p2_gg_httbar_abbrevd263h8, only: init_abbrevd263 => init_abbrev
      use p2_gg_httbar_diagramsh8l0, only: amplitudel0 => amplitude
      use p2_gg_httbar_groups
      implicit none
      real(ki), intent(in) :: scale2
      logical, intent(out) :: ok
      real(ki), intent(out) :: rational2
      complex(ki), dimension(numcs), intent(in), optional :: opt_amp0
      integer, dimension(numcs), intent(in), optional :: opt_perm
      real(ki), dimension(-2:0) :: samplitude

      real(ki), dimension(-2:0) :: acc
      real(ki), dimension(0:2,-2:0) :: samp_part

      logical :: acc_ok

      ok = .true.
      rational2 = 0.0_ki

      samplitude(:) = 0.0_ki
      if (present(opt_amp0)) then
         amp0 = opt_amp0
      else
         amp0 = amplitudel0()
      end if
      if (present(opt_perm)) then
         use_perm = .true.
         perm = opt_perm
      else
         use_perm = .false.
      end if

      rat2 = (0.0_ki, 0.0_ki)
      call init_lo()
        call init_abbrevd85()
        call init_abbrevd262()
        call init_abbrevd91()
        call init_abbrevd260()
        call init_abbrevd29()
        call init_abbrevd31()
        call init_abbrevd33()
        call init_abbrevd74()
        call init_abbrevd77()
        call init_abbrevd82()
        call init_abbrevd90()
        call init_abbrevd148()
        call init_abbrevd163()
        call init_abbrevd258()
        call init_abbrevd1()
        call init_abbrevd2()
        call init_abbrevd3()
        call init_abbrevd5()
        call init_abbrevd6()
        call init_abbrevd7()
        call init_abbrevd11()
        call init_abbrevd13()
        call init_abbrevd26()
        call init_abbrevd28()
        call init_abbrevd35()
        call init_abbrevd46()
        call init_abbrevd66()
        call init_abbrevd71()
        call init_abbrevd80()
        call init_abbrevd84()
        call init_abbrevd88()
        call init_abbrevd133()
        call init_abbrevd178()
        call init_abbrevd195()
        call init_abbrevd256()
        call init_abbrevd10()
        call init_abbrevd34()
        call init_abbrevd36()
        call init_abbrevd38()
        call init_abbrevd67()
        call init_abbrevd79()
        call init_abbrevd83()
        call init_abbrevd130()
        call init_abbrevd132()
        call init_abbrevd254()
        call init_abbrevd69()
        call init_abbrevd78()
        call init_abbrevd125()
        call init_abbrevd259()
        call init_abbrevd30()
        call init_abbrevd42()
        call init_abbrevd73()
        call init_abbrevd81()
        call init_abbrevd162()
        call init_abbrevd257()
        call init_abbrevd12()
        call init_abbrevd22()
        call init_abbrevd32()
        call init_abbrevd37()
        call init_abbrevd40()
        call init_abbrevd50()
        call init_abbrevd68()
        call init_abbrevd76()
        call init_abbrevd89()
        call init_abbrevd101()
        call init_abbrevd129()
        call init_abbrevd147()
        call init_abbrevd172()
        call init_abbrevd203()
        call init_abbrevd253()
        call init_abbrevd27()
        call init_abbrevd44()
        call init_abbrevd70()
        call init_abbrevd87()
        call init_abbrevd113()
        call init_abbrevd142()
        call init_abbrevd177()
        call init_abbrevd255()
        call init_abbrevd72()
        call init_abbrevd75()
        call init_abbrevd261()
        call init_abbrevd86()
        call init_abbrevd264()
        call init_abbrevd92()
        call init_abbrevd263()
      epspow=0
      samplitude(-2) = 0.0_ki
      samplitude(-1) = 0.0_ki
      if(debug_nlo_diagrams) then
         write(logfile,'(A22,G24.16,A6,G24.16,A4)') &
         & "<result name='r2' re='", real(rat2, ki), &
         &                 "' im='", aimag(rat2), "' />"
      end if
      rational2 = 2.0_ki * real(rat2, ki)
      samplitude(0) = 2.0_ki * real(rat2, ki)
         call evaluate_group0(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group1(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group2(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group3(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group4(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group5(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group6(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group7(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group8(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group9(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group10(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
         call evaluate_group11(scale2, acc, acc_ok)
         ok = ok .and. acc_ok
         samplitude(:) = samplitude(:) + acc
   end function samplitude
   !---#] function samplitude:
!---#[ subroutine evaluate_group0:
subroutine     evaluate_group0(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group0
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='0'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group0
!---#] subroutine evaluate_group0:
!---#[ subroutine evaluate_group1:
subroutine     evaluate_group1(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group1
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='1'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group1
!---#] subroutine evaluate_group1:
!---#[ subroutine evaluate_group2:
subroutine     evaluate_group2(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group2
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='2'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group2
!---#] subroutine evaluate_group2:
!---#[ subroutine evaluate_group3:
subroutine     evaluate_group3(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group3
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='3'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group3
!---#] subroutine evaluate_group3:
!---#[ subroutine evaluate_group4:
subroutine     evaluate_group4(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group4
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='4'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group4
!---#] subroutine evaluate_group4:
!---#[ subroutine evaluate_group5:
subroutine     evaluate_group5(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group5
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='5'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group5
!---#] subroutine evaluate_group5:
!---#[ subroutine evaluate_group6:
subroutine     evaluate_group6(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group6
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='6'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group6
!---#] subroutine evaluate_group6:
!---#[ subroutine evaluate_group7:
subroutine     evaluate_group7(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group7
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='7'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group7
!---#] subroutine evaluate_group7:
!---#[ subroutine evaluate_group8:
subroutine     evaluate_group8(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group8
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='8'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group8
!---#] subroutine evaluate_group8:
!---#[ subroutine evaluate_group9:
subroutine     evaluate_group9(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group9
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='9'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group9
!---#] subroutine evaluate_group9:
!---#[ subroutine evaluate_group10:
subroutine     evaluate_group10(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group10
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='10'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group10
!---#] subroutine evaluate_group10:
!---#[ subroutine evaluate_group11:
subroutine     evaluate_group11(scale2,samplitude,ok)
   use p2_gg_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p2_gg_httbar_globalsl1, only: epspow
   use p2_gg_httbar_ninjah8, only: ninja_reduce => ninja_reduce_group11
   implicit none
   real(ki), intent(in) :: scale2
   logical, intent(out) :: ok
   real(ki), dimension(-2:0), intent(out) :: samplitude
   complex(ki_nin), dimension(-2:0) :: tot
   complex(ki_nin) :: totr

   if(debug_nlo_diagrams) then
      write(logfile,*) "<diagram-group index='11'>"
      write(logfile,*) "<param name='epspow' value='", epspow, "'/>"
   end if
   select case(reduction_interoperation)
   case(2) ! use Ninja only
      call ninja_reduce(real(scale2, ki_nin), tot, totr, ok)
      samplitude(:) = 2.0_ki * real(tot(:), ki)
   case default
      print*, "Your current choice of reduction_interoperation is", &
            & reduction_interoperation
      print*, "This choice is not valid for your current setup."
      print*, "* This code was generated without support for Samurai."
      print*, "* This code was generated with support for Ninja."
      print*, "* This code was generated without support for Golem95."
   end select

   if(debug_nlo_diagrams) then
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-finite' value='", samplitude(0), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-single' value='", samplitude(-1), "'/>"
      write(logfile,'(A33,E24.16,A3)') &
         & "<result kind='nlo-double' value='", samplitude(-2), "'/>"
      if(ok) then
         write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
      else
         write(logfile,'(A29)') "<flag name='ok' status='no'/>"
      end if
      write(logfile,*) "</diagram-group>"
   end if
end subroutine evaluate_group11
!---#] subroutine evaluate_group11:
end module p2_gg_httbar_amplitudeh8

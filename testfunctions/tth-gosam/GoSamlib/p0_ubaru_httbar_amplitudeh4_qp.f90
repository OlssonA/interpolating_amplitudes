module    p0_ubaru_httbar_amplitudeh4_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp, &
       & reduction_interoperation
   use p0_ubaru_httbar_color_qp, only: numcs
   use p0_ubaru_httbar_groups
   use quadninjago_module, only: ki_nin
   use p0_ubaru_httbar_ninjah4_qp
   
   implicit none
   private

   public :: finite_renormalisation, samplitude
contains
!---#[ function finite_renormalisation:
   function     finite_renormalisation(scale2) result(amp)
      use p0_ubaru_httbar_util_qp, only: square
      use p0_ubaru_httbar_color_qp, only: CF, CA
      use p0_ubaru_httbar_kinematics_qp, only: &
      & num_light_quarks, num_gluons
      use p0_ubaru_httbar_diagramsh4l0_qp, only: amplitudel0 => amplitude
      implicit none
      real(ki), intent(in) :: scale2
      real(ki) :: amp
      amp = 0.0_ki
   end function finite_renormalisation
   !---#] function finite_renormalisation:

   !---#[ function samplitude:
   function     samplitude(scale2,ok,rational2,opt_amp0,opt_perm)
      use p0_ubaru_httbar_config, only: include_eps_terms, include_eps2_terms, &
      & logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_globalsl1_qp, only: amp0,perm, use_perm, epspow
      use p0_ubaru_httbar_globalsh4_qp, &
     & only: init_lo, rat2
      use p0_ubaru_httbar_abbrevd1h4_qp, only: init_abbrevd1 => init_abbrev
      use p0_ubaru_httbar_abbrevd13h4_qp, only: init_abbrevd13 => init_abbrev
      use p0_ubaru_httbar_abbrevd43h4_qp, only: init_abbrevd43 => init_abbrev
      use p0_ubaru_httbar_abbrevd58h4_qp, only: init_abbrevd58 => init_abbrev
      use p0_ubaru_httbar_abbrevd3h4_qp, only: init_abbrevd3 => init_abbrev
      use p0_ubaru_httbar_abbrevd59h4_qp, only: init_abbrevd59 => init_abbrev
      use p0_ubaru_httbar_abbrevd64h4_qp, only: init_abbrevd64 => init_abbrev
      use p0_ubaru_httbar_abbrevd67h4_qp, only: init_abbrevd67 => init_abbrev
      use p0_ubaru_httbar_abbrevd84h4_qp, only: init_abbrevd84 => init_abbrev
      use p0_ubaru_httbar_abbrevd2h4_qp, only: init_abbrevd2 => init_abbrev
      use p0_ubaru_httbar_abbrevd4h4_qp, only: init_abbrevd4 => init_abbrev
      use p0_ubaru_httbar_abbrevd21h4_qp, only: init_abbrevd21 => init_abbrev
      use p0_ubaru_httbar_abbrevd22h4_qp, only: init_abbrevd22 => init_abbrev
      use p0_ubaru_httbar_abbrevd39h4_qp, only: init_abbrevd39 => init_abbrev
      use p0_ubaru_httbar_abbrevd57h4_qp, only: init_abbrevd57 => init_abbrev
      use p0_ubaru_httbar_abbrevd65h4_qp, only: init_abbrevd65 => init_abbrev
      use p0_ubaru_httbar_abbrevd66h4_qp, only: init_abbrevd66 => init_abbrev
      use p0_ubaru_httbar_abbrevd71h4_qp, only: init_abbrevd71 => init_abbrev
      use p0_ubaru_httbar_abbrevd72h4_qp, only: init_abbrevd72 => init_abbrev
      use p0_ubaru_httbar_abbrevd77h4_qp, only: init_abbrevd77 => init_abbrev
      use p0_ubaru_httbar_abbrevd83h4_qp, only: init_abbrevd83 => init_abbrev
      use p0_ubaru_httbar_diagramsh4l0_qp, only: amplitudel0 => amplitude
      use p0_ubaru_httbar_groups
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
        call init_abbrevd1()
        call init_abbrevd13()
        call init_abbrevd43()
        call init_abbrevd58()
        call init_abbrevd3()
        call init_abbrevd59()
        call init_abbrevd64()
        call init_abbrevd67()
        call init_abbrevd84()
        call init_abbrevd2()
        call init_abbrevd4()
        call init_abbrevd21()
        call init_abbrevd22()
        call init_abbrevd39()
        call init_abbrevd57()
        call init_abbrevd65()
        call init_abbrevd66()
        call init_abbrevd71()
        call init_abbrevd72()
        call init_abbrevd77()
        call init_abbrevd83()
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
   end function samplitude
   !---#] function samplitude:
!---#[ subroutine evaluate_group0:
subroutine     evaluate_group0(scale2,samplitude,ok)
   use p0_ubaru_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p0_ubaru_httbar_globalsl1_qp, only: epspow
   use p0_ubaru_httbar_ninjah4_qp, only: ninja_reduce => ninja_reduce_group0
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
   case(4) ! use QuadNinja only
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
   use p0_ubaru_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p0_ubaru_httbar_globalsl1_qp, only: epspow
   use p0_ubaru_httbar_ninjah4_qp, only: ninja_reduce => ninja_reduce_group1
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
   case(4) ! use QuadNinja only
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
   use p0_ubaru_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p0_ubaru_httbar_globalsl1_qp, only: epspow
   use p0_ubaru_httbar_ninjah4_qp, only: ninja_reduce => ninja_reduce_group2
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
   case(4) ! use QuadNinja only
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
   use p0_ubaru_httbar_config, only: &
      & logfile, debug_nlo_diagrams
   use p0_ubaru_httbar_globalsl1_qp, only: epspow
   use p0_ubaru_httbar_ninjah4_qp, only: ninja_reduce => ninja_reduce_group3
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
   case(4) ! use QuadNinja only
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
end module p0_ubaru_httbar_amplitudeh4_qp

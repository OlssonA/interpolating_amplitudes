!
module     p2_gg_httbar_ninjah4
   ! This file has been generated for ninja
   use ninjago_module, only: ki_nin
   use p2_gg_httbar_config
   implicit none
   private
   public :: ninja_reduce_group0
   public :: ninja_reduce_group1
   public :: ninja_reduce_group2
   public :: ninja_reduce_group3
   public :: ninja_reduce_group4
   public :: ninja_reduce_group5
   public :: ninja_reduce_group6
   public :: ninja_reduce_group7
   public :: ninja_reduce_group8
   public :: ninja_reduce_group9
   public :: ninja_reduce_group10
   public :: ninja_reduce_group11
contains
!---#[ reduce groups with ninja:
!-----#[ subroutine ninja_reduce_group0:
subroutine     ninja_reduce_group0(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d85h4l1, only: numerator_diagram85 => numerator_ninja
   use p2_gg_httbar_d85h4l121, only: numerator_tmu_diagram85 => numerator_tmu
   use p2_gg_httbar_d85h4l131, only: numerator_t3_diagram85 => numerator_t3
   use p2_gg_httbar_d85h4l132, only: numerator_t2_diagram85 => numerator_t2
   use p2_gg_httbar_d262h4l1, only: numerator_diagram262 => numerator_ninja
   use p2_gg_httbar_d262h4l121, only: numerator_tmu_diagram262 => numerator_tmu
   use p2_gg_httbar_d262h4l131, only: numerator_t3_diagram262 => numerator_t3
   use p2_gg_httbar_d262h4l132, only: numerator_t2_diagram262 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(-k2+k5, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(-k2, ki_nin)
   msq(3) = 0.0_ki_nin
   Vi(:,3) = real(0, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k4, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(-k3-k4, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(es34-es51-es12, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-es12+mH**2-es23-2.0_ki*mT**2+es45, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es51-mT**2, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(0.0_ki, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es34-mT**2, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='85'>"
         end if
         call ninja_diagram(numerator_diagram85, numerator_t3_diagram85, numera&
         &tor_t2_diagram85, &
          &  numerator_tmu_diagram85, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='262'>"
         end if
         call ninja_diagram(numerator_diagram262, numerator_t3_diagram262, nume&
         &rator_t2_diagram262, &
          &  numerator_tmu_diagram262, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group0
!-----#] subroutine ninja_reduce_group0:
!-----#[ subroutine ninja_reduce_group1:
subroutine     ninja_reduce_group1(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d91h4l1, only: numerator_diagram91 => numerator_ninja
   use p2_gg_httbar_d91h4l121, only: numerator_tmu_diagram91 => numerator_tmu
   use p2_gg_httbar_d91h4l131, only: numerator_t3_diagram91 => numerator_t3
   use p2_gg_httbar_d91h4l132, only: numerator_t2_diagram91 => numerator_t2
   use p2_gg_httbar_d260h4l1, only: numerator_diagram260 => numerator_ninja
   use p2_gg_httbar_d260h4l121, only: numerator_tmu_diagram260 => numerator_tmu
   use p2_gg_httbar_d260h4l131, only: numerator_t3_diagram260 => numerator_t3
   use p2_gg_httbar_d260h4l132, only: numerator_t2_diagram260 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(-k4, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = 0.0_ki_nin
   Vi(:,3) = real(-k2, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k2+k5, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(-k2+k3+k5, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+mH**2-es23+es45-es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es34-es51-es12, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es23-es51+mT**2-es45, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(0.0_ki, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='91'>"
         end if
         call ninja_diagram(numerator_diagram91, numerator_t3_diagram91, numera&
         &tor_t2_diagram91, &
          &  numerator_tmu_diagram91, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='260'>"
         end if
         call ninja_diagram(numerator_diagram260, numerator_t3_diagram260, nume&
         &rator_t2_diagram260, &
          &  numerator_tmu_diagram260, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group1
!-----#] subroutine ninja_reduce_group1:
!-----#[ subroutine ninja_reduce_group2:
subroutine     ninja_reduce_group2(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d29h4l1, only: numerator_diagram29 => numerator_ninja
   use p2_gg_httbar_d29h4l121, only: numerator_tmu_diagram29 => numerator_tmu
   use p2_gg_httbar_d29h4l131, only: numerator_t3_diagram29 => numerator_t3
   use p2_gg_httbar_d29h4l132, only: numerator_t2_diagram29 => numerator_t2
   use p2_gg_httbar_d31h4l1, only: numerator_diagram31 => numerator_ninja
   use p2_gg_httbar_d31h4l121, only: numerator_tmu_diagram31 => numerator_tmu
   use p2_gg_httbar_d31h4l131, only: numerator_t3_diagram31 => numerator_t3
   use p2_gg_httbar_d31h4l132, only: numerator_t2_diagram31 => numerator_t2
   use p2_gg_httbar_d33h4l1, only: numerator_diagram33 => numerator_ninja
   use p2_gg_httbar_d33h4l121, only: numerator_tmu_diagram33 => numerator_tmu
   use p2_gg_httbar_d33h4l131, only: numerator_t3_diagram33 => numerator_t3
   use p2_gg_httbar_d33h4l132, only: numerator_t2_diagram33 => numerator_t2
   use p2_gg_httbar_d74h4l1, only: numerator_diagram74 => numerator_ninja
   use p2_gg_httbar_d74h4l121, only: numerator_tmu_diagram74 => numerator_tmu
   use p2_gg_httbar_d74h4l131, only: numerator_t3_diagram74 => numerator_t3
   use p2_gg_httbar_d74h4l132, only: numerator_t2_diagram74 => numerator_t2
   use p2_gg_httbar_d77h4l1, only: numerator_diagram77 => numerator_ninja
   use p2_gg_httbar_d77h4l121, only: numerator_tmu_diagram77 => numerator_tmu
   use p2_gg_httbar_d77h4l131, only: numerator_t3_diagram77 => numerator_t3
   use p2_gg_httbar_d77h4l132, only: numerator_t2_diagram77 => numerator_t2
   use p2_gg_httbar_d82h4l1, only: numerator_diagram82 => numerator_ninja
   use p2_gg_httbar_d82h4l121, only: numerator_tmu_diagram82 => numerator_tmu
   use p2_gg_httbar_d82h4l131, only: numerator_t3_diagram82 => numerator_t3
   use p2_gg_httbar_d82h4l132, only: numerator_t2_diagram82 => numerator_t2
   use p2_gg_httbar_d90h4l1, only: numerator_diagram90 => numerator_ninja
   use p2_gg_httbar_d90h4l121, only: numerator_tmu_diagram90 => numerator_tmu
   use p2_gg_httbar_d90h4l131, only: numerator_t3_diagram90 => numerator_t3
   use p2_gg_httbar_d90h4l132, only: numerator_t2_diagram90 => numerator_t2
   use p2_gg_httbar_d148h4l1, only: numerator_diagram148 => numerator_ninja
   use p2_gg_httbar_d148h4l121, only: numerator_tmu_diagram148 => numerator_tmu
   use p2_gg_httbar_d148h4l131, only: numerator_t3_diagram148 => numerator_t3
   use p2_gg_httbar_d148h4l132, only: numerator_t2_diagram148 => numerator_t2
   use p2_gg_httbar_d163h4l1, only: numerator_diagram163 => numerator_ninja
   use p2_gg_httbar_d163h4l121, only: numerator_tmu_diagram163 => numerator_tmu
   use p2_gg_httbar_d163h4l131, only: numerator_t3_diagram163 => numerator_t3
   use p2_gg_httbar_d163h4l132, only: numerator_t2_diagram163 => numerator_t2
   use p2_gg_httbar_d258h4l1, only: numerator_diagram258 => numerator_ninja
   use p2_gg_httbar_d258h4l121, only: numerator_tmu_diagram258 => numerator_tmu
   use p2_gg_httbar_d258h4l131, only: numerator_t3_diagram258 => numerator_t3
   use p2_gg_httbar_d258h4l132, only: numerator_t2_diagram258 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = 0.0_ki_nin
   Vi(:,1) = real(-k3-k4-k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(-k3-k4, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k4, ki_nin)
   msq(4) = 0.0_ki_nin
   Vi(:,4) = real(0, ki_nin)
   msq(5) = 0.0_ki_nin
   Vi(:,5) = real(-k2, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(0.0_ki, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(0.0_ki, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es34-mT**2, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(-mT**2+es51, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(0.0_ki, ki_nin)
   s_mat(4,5) = real(0.0_ki, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(0.0_ki, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='29'>"
         end if
         call ninja_diagram(numerator_diagram29, numerator_t3_diagram29, numera&
         &tor_t2_diagram29, &
          &  numerator_tmu_diagram29, &
          & 5, 3, 2, (/2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='31'>"
         end if
         call ninja_diagram(numerator_diagram31, numerator_t3_diagram31, numera&
         &tor_t2_diagram31, &
          &  numerator_tmu_diagram31, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='33'>"
         end if
         call ninja_diagram(numerator_diagram33, numerator_t3_diagram33, numera&
         &tor_t2_diagram33, &
          &  numerator_tmu_diagram33, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='74'>"
         end if
         call ninja_diagram(numerator_diagram74, numerator_t3_diagram74, numera&
         &tor_t2_diagram74, &
          &  numerator_tmu_diagram74, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='77'>"
         end if
         call ninja_diagram(numerator_diagram77, numerator_t3_diagram77, numera&
         &tor_t2_diagram77, &
          &  numerator_tmu_diagram77, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='82'>"
         end if
         call ninja_diagram(numerator_diagram82, numerator_t3_diagram82, numera&
         &tor_t2_diagram82, &
          &  numerator_tmu_diagram82, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='90'>"
         end if
         call ninja_diagram(numerator_diagram90, numerator_t3_diagram90, numera&
         &tor_t2_diagram90, &
          &  numerator_tmu_diagram90, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='148'>"
         end if
         call ninja_diagram(numerator_diagram148, numerator_t3_diagram148, nume&
         &rator_t2_diagram148, &
          &  numerator_tmu_diagram148, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='163'>"
         end if
         call ninja_diagram(numerator_diagram163, numerator_t3_diagram163, nume&
         &rator_t2_diagram163, &
          &  numerator_tmu_diagram163, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='258'>"
         end if
         call ninja_diagram(numerator_diagram258, numerator_t3_diagram258, nume&
         &rator_t2_diagram258, &
          &  numerator_tmu_diagram258, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group2
!-----#] subroutine ninja_reduce_group2:
!-----#[ subroutine ninja_reduce_group3:
subroutine     ninja_reduce_group3(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d1h4l1, only: numerator_diagram1 => numerator_ninja
   use p2_gg_httbar_d1h4l121, only: numerator_tmu_diagram1 => numerator_tmu
   use p2_gg_httbar_d1h4l131, only: numerator_t3_diagram1 => numerator_t3
   use p2_gg_httbar_d1h4l132, only: numerator_t2_diagram1 => numerator_t2
   use p2_gg_httbar_d2h4l1, only: numerator_diagram2 => numerator_ninja
   use p2_gg_httbar_d2h4l121, only: numerator_tmu_diagram2 => numerator_tmu
   use p2_gg_httbar_d2h4l131, only: numerator_t3_diagram2 => numerator_t3
   use p2_gg_httbar_d2h4l132, only: numerator_t2_diagram2 => numerator_t2
   use p2_gg_httbar_d3h4l1, only: numerator_diagram3 => numerator_ninja
   use p2_gg_httbar_d3h4l121, only: numerator_tmu_diagram3 => numerator_tmu
   use p2_gg_httbar_d3h4l131, only: numerator_t3_diagram3 => numerator_t3
   use p2_gg_httbar_d3h4l132, only: numerator_t2_diagram3 => numerator_t2
   use p2_gg_httbar_d5h4l1, only: numerator_diagram5 => numerator_ninja
   use p2_gg_httbar_d5h4l121, only: numerator_tmu_diagram5 => numerator_tmu
   use p2_gg_httbar_d5h4l131, only: numerator_t3_diagram5 => numerator_t3
   use p2_gg_httbar_d5h4l132, only: numerator_t2_diagram5 => numerator_t2
   use p2_gg_httbar_d6h4l1, only: numerator_diagram6 => numerator_ninja
   use p2_gg_httbar_d6h4l121, only: numerator_tmu_diagram6 => numerator_tmu
   use p2_gg_httbar_d6h4l131, only: numerator_t3_diagram6 => numerator_t3
   use p2_gg_httbar_d6h4l132, only: numerator_t2_diagram6 => numerator_t2
   use p2_gg_httbar_d7h4l1, only: numerator_diagram7 => numerator_ninja
   use p2_gg_httbar_d7h4l121, only: numerator_tmu_diagram7 => numerator_tmu
   use p2_gg_httbar_d7h4l131, only: numerator_t3_diagram7 => numerator_t3
   use p2_gg_httbar_d7h4l132, only: numerator_t2_diagram7 => numerator_t2
   use p2_gg_httbar_d11h4l1, only: numerator_diagram11 => numerator_ninja
   use p2_gg_httbar_d11h4l121, only: numerator_tmu_diagram11 => numerator_tmu
   use p2_gg_httbar_d11h4l131, only: numerator_t3_diagram11 => numerator_t3
   use p2_gg_httbar_d11h4l132, only: numerator_t2_diagram11 => numerator_t2
   use p2_gg_httbar_d13h4l1, only: numerator_diagram13 => numerator_ninja
   use p2_gg_httbar_d13h4l121, only: numerator_tmu_diagram13 => numerator_tmu
   use p2_gg_httbar_d13h4l131, only: numerator_t3_diagram13 => numerator_t3
   use p2_gg_httbar_d13h4l132, only: numerator_t2_diagram13 => numerator_t2
   use p2_gg_httbar_d26h4l1, only: numerator_diagram26 => numerator_ninja
   use p2_gg_httbar_d26h4l121, only: numerator_tmu_diagram26 => numerator_tmu
   use p2_gg_httbar_d26h4l131, only: numerator_t3_diagram26 => numerator_t3
   use p2_gg_httbar_d26h4l132, only: numerator_t2_diagram26 => numerator_t2
   use p2_gg_httbar_d28h4l1, only: numerator_diagram28 => numerator_ninja
   use p2_gg_httbar_d28h4l121, only: numerator_tmu_diagram28 => numerator_tmu
   use p2_gg_httbar_d28h4l131, only: numerator_t3_diagram28 => numerator_t3
   use p2_gg_httbar_d28h4l132, only: numerator_t2_diagram28 => numerator_t2
   use p2_gg_httbar_d35h4l1, only: numerator_diagram35 => numerator_ninja
   use p2_gg_httbar_d35h4l121, only: numerator_tmu_diagram35 => numerator_tmu
   use p2_gg_httbar_d35h4l131, only: numerator_t3_diagram35 => numerator_t3
   use p2_gg_httbar_d35h4l132, only: numerator_t2_diagram35 => numerator_t2
   use p2_gg_httbar_d46h4l1, only: numerator_diagram46 => numerator_ninja
   use p2_gg_httbar_d46h4l121, only: numerator_tmu_diagram46 => numerator_tmu
   use p2_gg_httbar_d46h4l131, only: numerator_t3_diagram46 => numerator_t3
   use p2_gg_httbar_d46h4l132, only: numerator_t2_diagram46 => numerator_t2
   use p2_gg_httbar_d66h4l1, only: numerator_diagram66 => numerator_ninja
   use p2_gg_httbar_d66h4l121, only: numerator_tmu_diagram66 => numerator_tmu
   use p2_gg_httbar_d66h4l131, only: numerator_t3_diagram66 => numerator_t3
   use p2_gg_httbar_d66h4l132, only: numerator_t2_diagram66 => numerator_t2
   use p2_gg_httbar_d71h4l1, only: numerator_diagram71 => numerator_ninja
   use p2_gg_httbar_d71h4l121, only: numerator_tmu_diagram71 => numerator_tmu
   use p2_gg_httbar_d71h4l131, only: numerator_t3_diagram71 => numerator_t3
   use p2_gg_httbar_d71h4l132, only: numerator_t2_diagram71 => numerator_t2
   use p2_gg_httbar_d80h4l1, only: numerator_diagram80 => numerator_ninja
   use p2_gg_httbar_d80h4l121, only: numerator_tmu_diagram80 => numerator_tmu
   use p2_gg_httbar_d80h4l131, only: numerator_t3_diagram80 => numerator_t3
   use p2_gg_httbar_d80h4l132, only: numerator_t2_diagram80 => numerator_t2
   use p2_gg_httbar_d84h4l1, only: numerator_diagram84 => numerator_ninja
   use p2_gg_httbar_d84h4l121, only: numerator_tmu_diagram84 => numerator_tmu
   use p2_gg_httbar_d84h4l131, only: numerator_t3_diagram84 => numerator_t3
   use p2_gg_httbar_d84h4l132, only: numerator_t2_diagram84 => numerator_t2
   use p2_gg_httbar_d88h4l1, only: numerator_diagram88 => numerator_ninja
   use p2_gg_httbar_d88h4l121, only: numerator_tmu_diagram88 => numerator_tmu
   use p2_gg_httbar_d88h4l131, only: numerator_t3_diagram88 => numerator_t3
   use p2_gg_httbar_d88h4l132, only: numerator_t2_diagram88 => numerator_t2
   use p2_gg_httbar_d133h4l1, only: numerator_diagram133 => numerator_ninja
   use p2_gg_httbar_d133h4l121, only: numerator_tmu_diagram133 => numerator_tmu
   use p2_gg_httbar_d133h4l131, only: numerator_t3_diagram133 => numerator_t3
   use p2_gg_httbar_d133h4l132, only: numerator_t2_diagram133 => numerator_t2
   use p2_gg_httbar_d178h4l1, only: numerator_diagram178 => numerator_ninja
   use p2_gg_httbar_d178h4l121, only: numerator_tmu_diagram178 => numerator_tmu
   use p2_gg_httbar_d178h4l131, only: numerator_t3_diagram178 => numerator_t3
   use p2_gg_httbar_d178h4l132, only: numerator_t2_diagram178 => numerator_t2
   use p2_gg_httbar_d195h4l1, only: numerator_diagram195 => numerator_ninja
   use p2_gg_httbar_d195h4l121, only: numerator_tmu_diagram195 => numerator_tmu
   use p2_gg_httbar_d195h4l131, only: numerator_t3_diagram195 => numerator_t3
   use p2_gg_httbar_d195h4l132, only: numerator_t2_diagram195 => numerator_t2
   use p2_gg_httbar_d256h4l1, only: numerator_diagram256 => numerator_ninja
   use p2_gg_httbar_d256h4l121, only: numerator_tmu_diagram256 => numerator_tmu
   use p2_gg_httbar_d256h4l131, only: numerator_t3_diagram256 => numerator_t3
   use p2_gg_httbar_d256h4l132, only: numerator_t2_diagram256 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = 0.0_ki_nin
   Vi(:,1) = real(-k3-k4-k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(-k3-k5, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k5, ki_nin)
   msq(4) = 0.0_ki_nin
   Vi(:,4) = real(0, ki_nin)
   msq(5) = 0.0_ki_nin
   Vi(:,5) = real(-k2, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(0.0_ki, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(es34-mT**2, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(0.0_ki, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(mT**2-es45+es23-es51, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es34-es51-es12, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(0.0_ki, ki_nin)
   s_mat(4,5) = real(0.0_ki, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(0.0_ki, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='1'>"
         end if
         call ninja_diagram(numerator_diagram1, numerator_t3_diagram1, numerato&
         &r_t2_diagram1, &
          &  numerator_tmu_diagram1, &
          & 5, 3, 1, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='2'>"
         end if
         call ninja_diagram(numerator_diagram2, numerator_t3_diagram2, numerato&
         &r_t2_diagram2, &
          &  numerator_tmu_diagram2, &
          & 5, 3, 1, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='3'>"
         end if
         call ninja_diagram(numerator_diagram3, numerator_t3_diagram3, numerato&
         &r_t2_diagram3, &
          &  numerator_tmu_diagram3, &
          & 5, 2, 1, (/1,4/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='5'>"
         end if
         call ninja_diagram(numerator_diagram5, numerator_t3_diagram5, numerato&
         &r_t2_diagram5, &
          &  numerator_tmu_diagram5, &
          & 5, 4, 2, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='6'>"
         end if
         call ninja_diagram(numerator_diagram6, numerator_t3_diagram6, numerato&
         &r_t2_diagram6, &
          &  numerator_tmu_diagram6, &
          & 5, 2, 1, (/1,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='7'>"
         end if
         call ninja_diagram(numerator_diagram7, numerator_t3_diagram7, numerato&
         &r_t2_diagram7, &
          &  numerator_tmu_diagram7, &
          & 5, 2, 1, (/4,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='11'>"
         end if
         call ninja_diagram(numerator_diagram11, numerator_t3_diagram11, numera&
         &tor_t2_diagram11, &
          &  numerator_tmu_diagram11, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='13'>"
         end if
         call ninja_diagram(numerator_diagram13, numerator_t3_diagram13, numera&
         &tor_t2_diagram13, &
          &  numerator_tmu_diagram13, &
          & 5, 3, 2, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='26'>"
         end if
         call ninja_diagram(numerator_diagram26, numerator_t3_diagram26, numera&
         &tor_t2_diagram26, &
          &  numerator_tmu_diagram26, &
          & 5, 3, 2, (/2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='28'>"
         end if
         call ninja_diagram(numerator_diagram28, numerator_t3_diagram28, numera&
         &tor_t2_diagram28, &
          &  numerator_tmu_diagram28, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='35'>"
         end if
         call ninja_diagram(numerator_diagram35, numerator_t3_diagram35, numera&
         &tor_t2_diagram35, &
          &  numerator_tmu_diagram35, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='46'>"
         end if
         call ninja_diagram(numerator_diagram46, numerator_t3_diagram46, numera&
         &tor_t2_diagram46, &
          &  numerator_tmu_diagram46, &
          & 5, 2, 2, (/1,4/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  +  acc
         totr = totr  +  accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='66'>"
         end if
         call ninja_diagram(numerator_diagram66, numerator_t3_diagram66, numera&
         &tor_t2_diagram66, &
          &  numerator_tmu_diagram66, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='71'>"
         end if
         call ninja_diagram(numerator_diagram71, numerator_t3_diagram71, numera&
         &tor_t2_diagram71, &
          &  numerator_tmu_diagram71, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='80'>"
         end if
         call ninja_diagram(numerator_diagram80, numerator_t3_diagram80, numera&
         &tor_t2_diagram80, &
          &  numerator_tmu_diagram80, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='84'>"
         end if
         call ninja_diagram(numerator_diagram84, numerator_t3_diagram84, numera&
         &tor_t2_diagram84, &
          &  numerator_tmu_diagram84, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='88'>"
         end if
         call ninja_diagram(numerator_diagram88, numerator_t3_diagram88, numera&
         &tor_t2_diagram88, &
          &  numerator_tmu_diagram88, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='133'>"
         end if
         call ninja_diagram(numerator_diagram133, numerator_t3_diagram133, nume&
         &rator_t2_diagram133, &
          &  numerator_tmu_diagram133, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='178'>"
         end if
         call ninja_diagram(numerator_diagram178, numerator_t3_diagram178, nume&
         &rator_t2_diagram178, &
          &  numerator_tmu_diagram178, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='195'>"
         end if
         call ninja_diagram(numerator_diagram195, numerator_t3_diagram195, nume&
         &rator_t2_diagram195, &
          &  numerator_tmu_diagram195, &
          & 5, 3, 3, (/1,4,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  +  acc
         totr = totr  +  accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='256'>"
         end if
         call ninja_diagram(numerator_diagram256, numerator_t3_diagram256, nume&
         &rator_t2_diagram256, &
          &  numerator_tmu_diagram256, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group3
!-----#] subroutine ninja_reduce_group3:
!-----#[ subroutine ninja_reduce_group4:
subroutine     ninja_reduce_group4(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d10h4l1, only: numerator_diagram10 => numerator_ninja
   use p2_gg_httbar_d10h4l121, only: numerator_tmu_diagram10 => numerator_tmu
   use p2_gg_httbar_d10h4l131, only: numerator_t3_diagram10 => numerator_t3
   use p2_gg_httbar_d10h4l132, only: numerator_t2_diagram10 => numerator_t2
   use p2_gg_httbar_d34h4l1, only: numerator_diagram34 => numerator_ninja
   use p2_gg_httbar_d34h4l121, only: numerator_tmu_diagram34 => numerator_tmu
   use p2_gg_httbar_d34h4l131, only: numerator_t3_diagram34 => numerator_t3
   use p2_gg_httbar_d34h4l132, only: numerator_t2_diagram34 => numerator_t2
   use p2_gg_httbar_d36h4l1, only: numerator_diagram36 => numerator_ninja
   use p2_gg_httbar_d36h4l121, only: numerator_tmu_diagram36 => numerator_tmu
   use p2_gg_httbar_d36h4l131, only: numerator_t3_diagram36 => numerator_t3
   use p2_gg_httbar_d36h4l132, only: numerator_t2_diagram36 => numerator_t2
   use p2_gg_httbar_d38h4l1, only: numerator_diagram38 => numerator_ninja
   use p2_gg_httbar_d38h4l121, only: numerator_tmu_diagram38 => numerator_tmu
   use p2_gg_httbar_d38h4l131, only: numerator_t3_diagram38 => numerator_t3
   use p2_gg_httbar_d38h4l132, only: numerator_t2_diagram38 => numerator_t2
   use p2_gg_httbar_d67h4l1, only: numerator_diagram67 => numerator_ninja
   use p2_gg_httbar_d67h4l121, only: numerator_tmu_diagram67 => numerator_tmu
   use p2_gg_httbar_d67h4l131, only: numerator_t3_diagram67 => numerator_t3
   use p2_gg_httbar_d67h4l132, only: numerator_t2_diagram67 => numerator_t2
   use p2_gg_httbar_d79h4l1, only: numerator_diagram79 => numerator_ninja
   use p2_gg_httbar_d79h4l121, only: numerator_tmu_diagram79 => numerator_tmu
   use p2_gg_httbar_d79h4l131, only: numerator_t3_diagram79 => numerator_t3
   use p2_gg_httbar_d79h4l132, only: numerator_t2_diagram79 => numerator_t2
   use p2_gg_httbar_d83h4l1, only: numerator_diagram83 => numerator_ninja
   use p2_gg_httbar_d83h4l121, only: numerator_tmu_diagram83 => numerator_tmu
   use p2_gg_httbar_d83h4l131, only: numerator_t3_diagram83 => numerator_t3
   use p2_gg_httbar_d83h4l132, only: numerator_t2_diagram83 => numerator_t2
   use p2_gg_httbar_d130h4l1, only: numerator_diagram130 => numerator_ninja
   use p2_gg_httbar_d130h4l121, only: numerator_tmu_diagram130 => numerator_tmu
   use p2_gg_httbar_d130h4l131, only: numerator_t3_diagram130 => numerator_t3
   use p2_gg_httbar_d130h4l132, only: numerator_t2_diagram130 => numerator_t2
   use p2_gg_httbar_d132h4l1, only: numerator_diagram132 => numerator_ninja
   use p2_gg_httbar_d132h4l121, only: numerator_tmu_diagram132 => numerator_tmu
   use p2_gg_httbar_d132h4l131, only: numerator_t3_diagram132 => numerator_t3
   use p2_gg_httbar_d132h4l132, only: numerator_t2_diagram132 => numerator_t2
   use p2_gg_httbar_d254h4l1, only: numerator_diagram254 => numerator_ninja
   use p2_gg_httbar_d254h4l121, only: numerator_tmu_diagram254 => numerator_tmu
   use p2_gg_httbar_d254h4l131, only: numerator_t3_diagram254 => numerator_t3
   use p2_gg_httbar_d254h4l132, only: numerator_t2_diagram254 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(-k3-k4, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(-k4, ki_nin)
   msq(3) = 0.0_ki_nin
   Vi(:,3) = real(0, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(k5, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(-k2+k5, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(es34-mT**2, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(-2.0_ki*mT**2+mH**2-es23+es45-es12, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(0.0_ki, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es34-es51-es12, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='10'>"
         end if
         call ninja_diagram(numerator_diagram10, numerator_t3_diagram10, numera&
         &tor_t2_diagram10, &
          &  numerator_tmu_diagram10, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='34'>"
         end if
         call ninja_diagram(numerator_diagram34, numerator_t3_diagram34, numera&
         &tor_t2_diagram34, &
          &  numerator_tmu_diagram34, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='36'>"
         end if
         call ninja_diagram(numerator_diagram36, numerator_t3_diagram36, numera&
         &tor_t2_diagram36, &
          &  numerator_tmu_diagram36, &
          & 5, 2, 2, (/1,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='38'>"
         end if
         call ninja_diagram(numerator_diagram38, numerator_t3_diagram38, numera&
         &tor_t2_diagram38, &
          &  numerator_tmu_diagram38, &
          & 5, 2, 2, (/3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='67'>"
         end if
         call ninja_diagram(numerator_diagram67, numerator_t3_diagram67, numera&
         &tor_t2_diagram67, &
          &  numerator_tmu_diagram67, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='79'>"
         end if
         call ninja_diagram(numerator_diagram79, numerator_t3_diagram79, numera&
         &tor_t2_diagram79, &
          &  numerator_tmu_diagram79, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='83'>"
         end if
         call ninja_diagram(numerator_diagram83, numerator_t3_diagram83, numera&
         &tor_t2_diagram83, &
          &  numerator_tmu_diagram83, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='130'>"
         end if
         call ninja_diagram(numerator_diagram130, numerator_t3_diagram130, nume&
         &rator_t2_diagram130, &
          &  numerator_tmu_diagram130, &
          & 5, 3, 2, (/1,2,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='132'>"
         end if
         call ninja_diagram(numerator_diagram132, numerator_t3_diagram132, nume&
         &rator_t2_diagram132, &
          &  numerator_tmu_diagram132, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='254'>"
         end if
         call ninja_diagram(numerator_diagram254, numerator_t3_diagram254, nume&
         &rator_t2_diagram254, &
          &  numerator_tmu_diagram254, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group4
!-----#] subroutine ninja_reduce_group4:
!-----#[ subroutine ninja_reduce_group5:
subroutine     ninja_reduce_group5(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d69h4l1, only: numerator_diagram69 => numerator_ninja
   use p2_gg_httbar_d69h4l121, only: numerator_tmu_diagram69 => numerator_tmu
   use p2_gg_httbar_d69h4l131, only: numerator_t3_diagram69 => numerator_t3
   use p2_gg_httbar_d69h4l132, only: numerator_t2_diagram69 => numerator_t2
   use p2_gg_httbar_d78h4l1, only: numerator_diagram78 => numerator_ninja
   use p2_gg_httbar_d78h4l121, only: numerator_tmu_diagram78 => numerator_tmu
   use p2_gg_httbar_d78h4l131, only: numerator_t3_diagram78 => numerator_t3
   use p2_gg_httbar_d78h4l132, only: numerator_t2_diagram78 => numerator_t2
   use p2_gg_httbar_d125h4l1, only: numerator_diagram125 => numerator_ninja
   use p2_gg_httbar_d125h4l121, only: numerator_tmu_diagram125 => numerator_tmu
   use p2_gg_httbar_d125h4l131, only: numerator_t3_diagram125 => numerator_t3
   use p2_gg_httbar_d125h4l132, only: numerator_t2_diagram125 => numerator_t2
   use p2_gg_httbar_d259h4l1, only: numerator_diagram259 => numerator_ninja
   use p2_gg_httbar_d259h4l121, only: numerator_tmu_diagram259 => numerator_tmu
   use p2_gg_httbar_d259h4l131, only: numerator_t3_diagram259 => numerator_t3
   use p2_gg_httbar_d259h4l132, only: numerator_t2_diagram259 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(-k4, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(k5, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k2+k5, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(-k2+k3+k5, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+mH**2-es23+es45-es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es34-es51-es12, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es23-es51+mT**2-es45, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es23-2.0_ki*mT**2, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='69'>"
         end if
         call ninja_diagram(numerator_diagram69, numerator_t3_diagram69, numera&
         &tor_t2_diagram69, &
          &  numerator_tmu_diagram69, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='78'>"
         end if
         call ninja_diagram(numerator_diagram78, numerator_t3_diagram78, numera&
         &tor_t2_diagram78, &
          &  numerator_tmu_diagram78, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='125'>"
         end if
         call ninja_diagram(numerator_diagram125, numerator_t3_diagram125, nume&
         &rator_t2_diagram125, &
          &  numerator_tmu_diagram125, &
          & 5, 4, 4, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='259'>"
         end if
         call ninja_diagram(numerator_diagram259, numerator_t3_diagram259, nume&
         &rator_t2_diagram259, &
          &  numerator_tmu_diagram259, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group5
!-----#] subroutine ninja_reduce_group5:
!-----#[ subroutine ninja_reduce_group6:
subroutine     ninja_reduce_group6(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d30h4l1, only: numerator_diagram30 => numerator_ninja
   use p2_gg_httbar_d30h4l121, only: numerator_tmu_diagram30 => numerator_tmu
   use p2_gg_httbar_d30h4l131, only: numerator_t3_diagram30 => numerator_t3
   use p2_gg_httbar_d30h4l132, only: numerator_t2_diagram30 => numerator_t2
   use p2_gg_httbar_d42h4l1, only: numerator_diagram42 => numerator_ninja
   use p2_gg_httbar_d42h4l121, only: numerator_tmu_diagram42 => numerator_tmu
   use p2_gg_httbar_d42h4l131, only: numerator_t3_diagram42 => numerator_t3
   use p2_gg_httbar_d42h4l132, only: numerator_t2_diagram42 => numerator_t2
   use p2_gg_httbar_d73h4l1, only: numerator_diagram73 => numerator_ninja
   use p2_gg_httbar_d73h4l121, only: numerator_tmu_diagram73 => numerator_tmu
   use p2_gg_httbar_d73h4l131, only: numerator_t3_diagram73 => numerator_t3
   use p2_gg_httbar_d73h4l132, only: numerator_t2_diagram73 => numerator_t2
   use p2_gg_httbar_d81h4l1, only: numerator_diagram81 => numerator_ninja
   use p2_gg_httbar_d81h4l121, only: numerator_tmu_diagram81 => numerator_tmu
   use p2_gg_httbar_d81h4l131, only: numerator_t3_diagram81 => numerator_t3
   use p2_gg_httbar_d81h4l132, only: numerator_t2_diagram81 => numerator_t2
   use p2_gg_httbar_d162h4l1, only: numerator_diagram162 => numerator_ninja
   use p2_gg_httbar_d162h4l121, only: numerator_tmu_diagram162 => numerator_tmu
   use p2_gg_httbar_d162h4l131, only: numerator_t3_diagram162 => numerator_t3
   use p2_gg_httbar_d162h4l132, only: numerator_t2_diagram162 => numerator_t2
   use p2_gg_httbar_d257h4l1, only: numerator_diagram257 => numerator_ninja
   use p2_gg_httbar_d257h4l121, only: numerator_tmu_diagram257 => numerator_tmu
   use p2_gg_httbar_d257h4l131, only: numerator_t3_diagram257 => numerator_t3
   use p2_gg_httbar_d257h4l132, only: numerator_t2_diagram257 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(k5, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k4, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k3-k4, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(k2-k3-k4, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es34-mT**2, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es51-mT**2, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es23-2.0_ki*mT**2, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='30'>"
         end if
         call ninja_diagram(numerator_diagram30, numerator_t3_diagram30, numera&
         &tor_t2_diagram30, &
          &  numerator_tmu_diagram30, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='42'>"
         end if
         call ninja_diagram(numerator_diagram42, numerator_t3_diagram42, numera&
         &tor_t2_diagram42, &
          &  numerator_tmu_diagram42, &
          & 5, 2, 2, (/2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='73'>"
         end if
         call ninja_diagram(numerator_diagram73, numerator_t3_diagram73, numera&
         &tor_t2_diagram73, &
          &  numerator_tmu_diagram73, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='81'>"
         end if
         call ninja_diagram(numerator_diagram81, numerator_t3_diagram81, numera&
         &tor_t2_diagram81, &
          &  numerator_tmu_diagram81, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='162'>"
         end if
         call ninja_diagram(numerator_diagram162, numerator_t3_diagram162, nume&
         &rator_t2_diagram162, &
          &  numerator_tmu_diagram162, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='257'>"
         end if
         call ninja_diagram(numerator_diagram257, numerator_t3_diagram257, nume&
         &rator_t2_diagram257, &
          &  numerator_tmu_diagram257, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group6
!-----#] subroutine ninja_reduce_group6:
!-----#[ subroutine ninja_reduce_group7:
subroutine     ninja_reduce_group7(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d12h4l1, only: numerator_diagram12 => numerator_ninja
   use p2_gg_httbar_d12h4l121, only: numerator_tmu_diagram12 => numerator_tmu
   use p2_gg_httbar_d12h4l131, only: numerator_t3_diagram12 => numerator_t3
   use p2_gg_httbar_d12h4l132, only: numerator_t2_diagram12 => numerator_t2
   use p2_gg_httbar_d22h4l1, only: numerator_diagram22 => numerator_ninja
   use p2_gg_httbar_d22h4l121, only: numerator_tmu_diagram22 => numerator_tmu
   use p2_gg_httbar_d22h4l131, only: numerator_t3_diagram22 => numerator_t3
   use p2_gg_httbar_d22h4l132, only: numerator_t2_diagram22 => numerator_t2
   use p2_gg_httbar_d32h4l1, only: numerator_diagram32 => numerator_ninja
   use p2_gg_httbar_d32h4l121, only: numerator_tmu_diagram32 => numerator_tmu
   use p2_gg_httbar_d32h4l131, only: numerator_t3_diagram32 => numerator_t3
   use p2_gg_httbar_d32h4l132, only: numerator_t2_diagram32 => numerator_t2
   use p2_gg_httbar_d37h4l1, only: numerator_diagram37 => numerator_ninja
   use p2_gg_httbar_d37h4l121, only: numerator_tmu_diagram37 => numerator_tmu
   use p2_gg_httbar_d37h4l131, only: numerator_t3_diagram37 => numerator_t3
   use p2_gg_httbar_d37h4l132, only: numerator_t2_diagram37 => numerator_t2
   use p2_gg_httbar_d40h4l1, only: numerator_diagram40 => numerator_ninja
   use p2_gg_httbar_d40h4l121, only: numerator_tmu_diagram40 => numerator_tmu
   use p2_gg_httbar_d40h4l131, only: numerator_t3_diagram40 => numerator_t3
   use p2_gg_httbar_d40h4l132, only: numerator_t2_diagram40 => numerator_t2
   use p2_gg_httbar_d50h4l1, only: numerator_diagram50 => numerator_ninja
   use p2_gg_httbar_d50h4l121, only: numerator_tmu_diagram50 => numerator_tmu
   use p2_gg_httbar_d50h4l131, only: numerator_t3_diagram50 => numerator_t3
   use p2_gg_httbar_d50h4l132, only: numerator_t2_diagram50 => numerator_t2
   use p2_gg_httbar_d68h4l1, only: numerator_diagram68 => numerator_ninja
   use p2_gg_httbar_d68h4l121, only: numerator_tmu_diagram68 => numerator_tmu
   use p2_gg_httbar_d68h4l131, only: numerator_t3_diagram68 => numerator_t3
   use p2_gg_httbar_d68h4l132, only: numerator_t2_diagram68 => numerator_t2
   use p2_gg_httbar_d76h4l1, only: numerator_diagram76 => numerator_ninja
   use p2_gg_httbar_d76h4l121, only: numerator_tmu_diagram76 => numerator_tmu
   use p2_gg_httbar_d76h4l131, only: numerator_t3_diagram76 => numerator_t3
   use p2_gg_httbar_d76h4l132, only: numerator_t2_diagram76 => numerator_t2
   use p2_gg_httbar_d89h4l1, only: numerator_diagram89 => numerator_ninja
   use p2_gg_httbar_d89h4l121, only: numerator_tmu_diagram89 => numerator_tmu
   use p2_gg_httbar_d89h4l131, only: numerator_t3_diagram89 => numerator_t3
   use p2_gg_httbar_d89h4l132, only: numerator_t2_diagram89 => numerator_t2
   use p2_gg_httbar_d101h4l1, only: numerator_diagram101 => numerator_ninja
   use p2_gg_httbar_d101h4l121, only: numerator_tmu_diagram101 => numerator_tmu
   use p2_gg_httbar_d101h4l131, only: numerator_t3_diagram101 => numerator_t3
   use p2_gg_httbar_d101h4l132, only: numerator_t2_diagram101 => numerator_t2
   use p2_gg_httbar_d129h4l1, only: numerator_diagram129 => numerator_ninja
   use p2_gg_httbar_d129h4l121, only: numerator_tmu_diagram129 => numerator_tmu
   use p2_gg_httbar_d129h4l131, only: numerator_t3_diagram129 => numerator_t3
   use p2_gg_httbar_d129h4l132, only: numerator_t2_diagram129 => numerator_t2
   use p2_gg_httbar_d147h4l1, only: numerator_diagram147 => numerator_ninja
   use p2_gg_httbar_d147h4l121, only: numerator_tmu_diagram147 => numerator_tmu
   use p2_gg_httbar_d147h4l131, only: numerator_t3_diagram147 => numerator_t3
   use p2_gg_httbar_d147h4l132, only: numerator_t2_diagram147 => numerator_t2
   use p2_gg_httbar_d172h4l1, only: numerator_diagram172 => numerator_ninja
   use p2_gg_httbar_d172h4l121, only: numerator_tmu_diagram172 => numerator_tmu
   use p2_gg_httbar_d172h4l131, only: numerator_t3_diagram172 => numerator_t3
   use p2_gg_httbar_d172h4l132, only: numerator_t2_diagram172 => numerator_t2
   use p2_gg_httbar_d203h4l1, only: numerator_diagram203 => numerator_ninja
   use p2_gg_httbar_d203h4l121, only: numerator_tmu_diagram203 => numerator_tmu
   use p2_gg_httbar_d203h4l131, only: numerator_t3_diagram203 => numerator_t3
   use p2_gg_httbar_d203h4l132, only: numerator_t2_diagram203 => numerator_t2
   use p2_gg_httbar_d253h4l1, only: numerator_diagram253 => numerator_ninja
   use p2_gg_httbar_d253h4l121, only: numerator_tmu_diagram253 => numerator_tmu
   use p2_gg_httbar_d253h4l131, only: numerator_t3_diagram253 => numerator_t3
   use p2_gg_httbar_d253h4l132, only: numerator_t2_diagram253 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(k3+k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(k5, ki_nin)
   msq(3) = 0.0_ki_nin
   Vi(:,3) = real(0, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k4, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(k2-k4, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es12-2.0_ki*mT**2, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(-2.0_ki*mT**2-es12+es45+mH**2-es23, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(0.0_ki, ki_nin)
   s_mat(3,4) = real(0.0_ki, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='12'>"
         end if
         call ninja_diagram(numerator_diagram12, numerator_t3_diagram12, numera&
         &tor_t2_diagram12, &
          &  numerator_tmu_diagram12, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='22'>"
         end if
         call ninja_diagram(numerator_diagram22, numerator_t3_diagram22, numera&
         &tor_t2_diagram22, &
          &  numerator_tmu_diagram22, &
          & 5, 3, 3, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='32'>"
         end if
         call ninja_diagram(numerator_diagram32, numerator_t3_diagram32, numera&
         &tor_t2_diagram32, &
          &  numerator_tmu_diagram32, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='37'>"
         end if
         call ninja_diagram(numerator_diagram37, numerator_t3_diagram37, numera&
         &tor_t2_diagram37, &
          &  numerator_tmu_diagram37, &
          & 5, 2, 2, (/1,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='40'>"
         end if
         call ninja_diagram(numerator_diagram40, numerator_t3_diagram40, numera&
         &tor_t2_diagram40, &
          &  numerator_tmu_diagram40, &
          & 5, 2, 2, (/3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='50'>"
         end if
         call ninja_diagram(numerator_diagram50, numerator_t3_diagram50, numera&
         &tor_t2_diagram50, &
          &  numerator_tmu_diagram50, &
          & 5, 2, 2, (/1,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='68'>"
         end if
         call ninja_diagram(numerator_diagram68, numerator_t3_diagram68, numera&
         &tor_t2_diagram68, &
          &  numerator_tmu_diagram68, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='76'>"
         end if
         call ninja_diagram(numerator_diagram76, numerator_t3_diagram76, numera&
         &tor_t2_diagram76, &
          &  numerator_tmu_diagram76, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='89'>"
         end if
         call ninja_diagram(numerator_diagram89, numerator_t3_diagram89, numera&
         &tor_t2_diagram89, &
          &  numerator_tmu_diagram89, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='101'>"
         end if
         call ninja_diagram(numerator_diagram101, numerator_t3_diagram101, nume&
         &rator_t2_diagram101, &
          &  numerator_tmu_diagram101, &
          & 5, 4, 4, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='129'>"
         end if
         call ninja_diagram(numerator_diagram129, numerator_t3_diagram129, nume&
         &rator_t2_diagram129, &
          &  numerator_tmu_diagram129, &
          & 5, 3, 2, (/1,2,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='147'>"
         end if
         call ninja_diagram(numerator_diagram147, numerator_t3_diagram147, nume&
         &rator_t2_diagram147, &
          &  numerator_tmu_diagram147, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='172'>"
         end if
         call ninja_diagram(numerator_diagram172, numerator_t3_diagram172, nume&
         &rator_t2_diagram172, &
          &  numerator_tmu_diagram172, &
          & 5, 3, 3, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='203'>"
         end if
         call ninja_diagram(numerator_diagram203, numerator_t3_diagram203, nume&
         &rator_t2_diagram203, &
          &  numerator_tmu_diagram203, &
          & 5, 3, 3, (/1,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='253'>"
         end if
         call ninja_diagram(numerator_diagram253, numerator_t3_diagram253, nume&
         &rator_t2_diagram253, &
          &  numerator_tmu_diagram253, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group7
!-----#] subroutine ninja_reduce_group7:
!-----#[ subroutine ninja_reduce_group8:
subroutine     ninja_reduce_group8(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d27h4l1, only: numerator_diagram27 => numerator_ninja
   use p2_gg_httbar_d27h4l121, only: numerator_tmu_diagram27 => numerator_tmu
   use p2_gg_httbar_d27h4l131, only: numerator_t3_diagram27 => numerator_t3
   use p2_gg_httbar_d27h4l132, only: numerator_t2_diagram27 => numerator_t2
   use p2_gg_httbar_d44h4l1, only: numerator_diagram44 => numerator_ninja
   use p2_gg_httbar_d44h4l121, only: numerator_tmu_diagram44 => numerator_tmu
   use p2_gg_httbar_d44h4l131, only: numerator_t3_diagram44 => numerator_t3
   use p2_gg_httbar_d44h4l132, only: numerator_t2_diagram44 => numerator_t2
   use p2_gg_httbar_d70h4l1, only: numerator_diagram70 => numerator_ninja
   use p2_gg_httbar_d70h4l121, only: numerator_tmu_diagram70 => numerator_tmu
   use p2_gg_httbar_d70h4l131, only: numerator_t3_diagram70 => numerator_t3
   use p2_gg_httbar_d70h4l132, only: numerator_t2_diagram70 => numerator_t2
   use p2_gg_httbar_d87h4l1, only: numerator_diagram87 => numerator_ninja
   use p2_gg_httbar_d87h4l121, only: numerator_tmu_diagram87 => numerator_tmu
   use p2_gg_httbar_d87h4l131, only: numerator_t3_diagram87 => numerator_t3
   use p2_gg_httbar_d87h4l132, only: numerator_t2_diagram87 => numerator_t2
   use p2_gg_httbar_d113h4l1, only: numerator_diagram113 => numerator_ninja
   use p2_gg_httbar_d113h4l121, only: numerator_tmu_diagram113 => numerator_tmu
   use p2_gg_httbar_d113h4l131, only: numerator_t3_diagram113 => numerator_t3
   use p2_gg_httbar_d113h4l132, only: numerator_t2_diagram113 => numerator_t2
   use p2_gg_httbar_d142h4l1, only: numerator_diagram142 => numerator_ninja
   use p2_gg_httbar_d142h4l121, only: numerator_tmu_diagram142 => numerator_tmu
   use p2_gg_httbar_d142h4l131, only: numerator_t3_diagram142 => numerator_t3
   use p2_gg_httbar_d142h4l132, only: numerator_t2_diagram142 => numerator_t2
   use p2_gg_httbar_d177h4l1, only: numerator_diagram177 => numerator_ninja
   use p2_gg_httbar_d177h4l121, only: numerator_tmu_diagram177 => numerator_tmu
   use p2_gg_httbar_d177h4l131, only: numerator_t3_diagram177 => numerator_t3
   use p2_gg_httbar_d177h4l132, only: numerator_t2_diagram177 => numerator_t2
   use p2_gg_httbar_d255h4l1, only: numerator_diagram255 => numerator_ninja
   use p2_gg_httbar_d255h4l121, only: numerator_tmu_diagram255 => numerator_tmu
   use p2_gg_httbar_d255h4l131, only: numerator_t3_diagram255 => numerator_t3
   use p2_gg_httbar_d255h4l132, only: numerator_t2_diagram255 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 5

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(-k4, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(k5, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(k3+k5, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(-k2+k3+k5, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2+es12, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es23-es51+mT**2-es45, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es23-2.0_ki*mT**2, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='27'>"
         end if
         call ninja_diagram(numerator_diagram27, numerator_t3_diagram27, numera&
         &tor_t2_diagram27, &
          &  numerator_tmu_diagram27, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='44'>"
         end if
         call ninja_diagram(numerator_diagram44, numerator_t3_diagram44, numera&
         &tor_t2_diagram44, &
          &  numerator_tmu_diagram44, &
          & 5, 2, 2, (/2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='70'>"
         end if
         call ninja_diagram(numerator_diagram70, numerator_t3_diagram70, numera&
         &tor_t2_diagram70, &
          &  numerator_tmu_diagram70, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='87'>"
         end if
         call ninja_diagram(numerator_diagram87, numerator_t3_diagram87, numera&
         &tor_t2_diagram87, &
          &  numerator_tmu_diagram87, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='113'>"
         end if
         call ninja_diagram(numerator_diagram113, numerator_t3_diagram113, nume&
         &rator_t2_diagram113, &
          &  numerator_tmu_diagram113, &
          & 5, 4, 4, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='142'>"
         end if
         call ninja_diagram(numerator_diagram142, numerator_t3_diagram142, nume&
         &rator_t2_diagram142, &
          &  numerator_tmu_diagram142, &
          & 5, 3, 3, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", -real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", -real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", -real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", -real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='177'>"
         end if
         call ninja_diagram(numerator_diagram177, numerator_t3_diagram177, nume&
         &rator_t2_diagram177, &
          &  numerator_tmu_diagram177, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='255'>"
         end if
         call ninja_diagram(numerator_diagram255, numerator_t3_diagram255, nume&
         &rator_t2_diagram255, &
          &  numerator_tmu_diagram255, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group8
!-----#] subroutine ninja_reduce_group8:
!-----#[ subroutine ninja_reduce_group9:
subroutine     ninja_reduce_group9(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d72h4l1, only: numerator_diagram72 => numerator_ninja
   use p2_gg_httbar_d72h4l121, only: numerator_tmu_diagram72 => numerator_tmu
   use p2_gg_httbar_d72h4l131, only: numerator_t3_diagram72 => numerator_t3
   use p2_gg_httbar_d72h4l132, only: numerator_t2_diagram72 => numerator_t2
   use p2_gg_httbar_d75h4l1, only: numerator_diagram75 => numerator_ninja
   use p2_gg_httbar_d75h4l121, only: numerator_tmu_diagram75 => numerator_tmu
   use p2_gg_httbar_d75h4l131, only: numerator_t3_diagram75 => numerator_t3
   use p2_gg_httbar_d75h4l132, only: numerator_t2_diagram75 => numerator_t2
   use p2_gg_httbar_d261h4l1, only: numerator_diagram261 => numerator_ninja
   use p2_gg_httbar_d261h4l121, only: numerator_tmu_diagram261 => numerator_tmu
   use p2_gg_httbar_d261h4l131, only: numerator_t3_diagram261 => numerator_t3
   use p2_gg_httbar_d261h4l132, only: numerator_t2_diagram261 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = real(mT*mT, ki_nin)
   Vi(:,1) = real(k5, ki_nin)
   msq(2) = 0.0_ki_nin
   Vi(:,2) = real(0, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k4, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(k2-k4, ki_nin)
   msq(5) = real(mT*mT, ki_nin)
   Vi(:,5) = real(k2-k3-k4, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(-2.0_ki*mT**2+es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(-2.0_ki*mT**2-es12+es45+mH**2-es23, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(0.0_ki, ki_nin)
   s_mat(2,3) = real(0.0_ki, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es51-mT**2, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es23-2.0_ki*mT**2, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(-2.0_ki*mT**2, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='72'>"
         end if
         call ninja_diagram(numerator_diagram72, numerator_t3_diagram72, numera&
         &tor_t2_diagram72, &
          &  numerator_tmu_diagram72, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='75'>"
         end if
         call ninja_diagram(numerator_diagram75, numerator_t3_diagram75, numera&
         &tor_t2_diagram75, &
          &  numerator_tmu_diagram75, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='261'>"
         end if
         call ninja_diagram(numerator_diagram261, numerator_t3_diagram261, nume&
         &rator_t2_diagram261, &
          &  numerator_tmu_diagram261, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group9
!-----#] subroutine ninja_reduce_group9:
!-----#[ subroutine ninja_reduce_group10:
subroutine     ninja_reduce_group10(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d86h4l1, only: numerator_diagram86 => numerator_ninja
   use p2_gg_httbar_d86h4l121, only: numerator_tmu_diagram86 => numerator_tmu
   use p2_gg_httbar_d86h4l131, only: numerator_t3_diagram86 => numerator_t3
   use p2_gg_httbar_d86h4l132, only: numerator_t2_diagram86 => numerator_t2
   use p2_gg_httbar_d264h4l1, only: numerator_diagram264 => numerator_ninja
   use p2_gg_httbar_d264h4l121, only: numerator_tmu_diagram264 => numerator_tmu
   use p2_gg_httbar_d264h4l131, only: numerator_t3_diagram264 => numerator_t3
   use p2_gg_httbar_d264h4l132, only: numerator_t2_diagram264 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = 0.0_ki_nin
   Vi(:,1) = real(k2-k3-k4-k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(k2-k3-k4, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(-k3-k4, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k4, ki_nin)
   msq(5) = 0.0_ki_nin
   Vi(:,5) = real(0, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(0.0_ki, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(es34-es51-es12, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es23-es51+mT**2-es45, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(0.0_ki, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es23-2.0_ki*mT**2, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es51-mT**2, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es34-mT**2, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(0.0_ki, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(0.0_ki, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='86'>"
         end if
         call ninja_diagram(numerator_diagram86, numerator_t3_diagram86, numera&
         &tor_t2_diagram86, &
          &  numerator_tmu_diagram86, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='264'>"
         end if
         call ninja_diagram(numerator_diagram264, numerator_t3_diagram264, nume&
         &rator_t2_diagram264, &
          &  numerator_tmu_diagram264, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group10
!-----#] subroutine ninja_reduce_group10:
!-----#[ subroutine ninja_reduce_group11:
subroutine     ninja_reduce_group11(scale2,tot,totr,ok)
   use iso_c_binding, only: c_ptr, c_loc, c_int
   use ninjago_module
   use p2_gg_httbar_kinematics
   use p2_gg_httbar_model
   use p2_gg_httbar_d92h4l1, only: numerator_diagram92 => numerator_ninja
   use p2_gg_httbar_d92h4l121, only: numerator_tmu_diagram92 => numerator_tmu
   use p2_gg_httbar_d92h4l131, only: numerator_t3_diagram92 => numerator_t3
   use p2_gg_httbar_d92h4l132, only: numerator_t2_diagram92 => numerator_t2
   use p2_gg_httbar_d263h4l1, only: numerator_diagram263 => numerator_ninja
   use p2_gg_httbar_d263h4l121, only: numerator_tmu_diagram263 => numerator_tmu
   use p2_gg_httbar_d263h4l131, only: numerator_t3_diagram263 => numerator_t3
   use p2_gg_httbar_d263h4l132, only: numerator_t2_diagram263 => numerator_t2
   use p2_gg_httbar_globalsl1, only: epspow

   implicit none
   real(ki_nin), intent(in) :: scale2
   complex(ki_nin), dimension(-2:0), intent(out) :: tot
   complex(ki_nin), intent(out) :: totr
   logical, intent(out) :: ok

   complex(ki_nin), dimension(-2:0) :: acc
   complex(ki_nin) :: accr
   integer(c_int) :: acc_ok

   integer :: istopm, istop0

   integer, parameter :: effective_group_rank = 4

   !-----------#[ invariants for ninja:
   real(ki_nin), dimension(5,5) :: s_mat
   !-----------#] initialize invariants:
   real(ki_nin), dimension(5) :: msq
   real(ki_nin), dimension(4,5) :: Vi

   ok = .true.

   if (ninja_test.eq.1) then
      istopm = 1
      istop0 = 1
   else
      istopm = ninja_istop
      istop0 = max(2,ninja_istop)
   end if
   msq(1) = 0.0_ki_nin
   Vi(:,1) = real(k2-k3-k4-k5, ki_nin)
   msq(2) = real(mT*mT, ki_nin)
   Vi(:,2) = real(k2-k3-k4, ki_nin)
   msq(3) = real(mT*mT, ki_nin)
   Vi(:,3) = real(k2-k4, ki_nin)
   msq(4) = real(mT*mT, ki_nin)
   Vi(:,4) = real(-k4, ki_nin)
   msq(5) = 0.0_ki_nin
   Vi(:,5) = real(0, ki_nin)
   !-----------#[ initialize invariants:
   s_mat(1,1) = real(0.0_ki, ki_nin)
   s_mat(1,2) = real(0.0_ki, ki_nin)
   s_mat(2,1) = s_mat(1,2)
   s_mat(1,3) = real(mH**2+es12+mT**2-es34-es45, ki_nin)
   s_mat(3,1) = s_mat(1,3)
   s_mat(1,4) = real(es23-es51+mT**2-es45, ki_nin)
   s_mat(4,1) = s_mat(1,4)
   s_mat(1,5) = real(0.0_ki, ki_nin)
   s_mat(5,1) = s_mat(1,5)
   s_mat(2,2) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(2,3) = real(mH**2-2.0_ki*mT**2, ki_nin)
   s_mat(3,2) = s_mat(2,3)
   s_mat(2,4) = real(es23-2.0_ki*mT**2, ki_nin)
   s_mat(4,2) = s_mat(2,4)
   s_mat(2,5) = real(es51-mT**2, ki_nin)
   s_mat(5,2) = s_mat(2,5)
   s_mat(3,3) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(3,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,3) = s_mat(3,4)
   s_mat(3,5) = real(es51+mH**2-es23-es34, ki_nin)
   s_mat(5,3) = s_mat(3,5)
   s_mat(4,4) = real(-2.0_ki*mT**2, ki_nin)
   s_mat(4,5) = real(0.0_ki, ki_nin)
   s_mat(5,4) = s_mat(4,5)
   s_mat(5,5) = real(0.0_ki, ki_nin)
   !-----------#] initialize invariants


      !------#[ sum over reduction of single diagrams:
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='92'>"
         end if
         call ninja_diagram(numerator_diagram92, numerator_t3_diagram92, numera&
         &tor_t2_diagram92, &
          &  numerator_tmu_diagram92, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot =  + acc
         totr =  + accr
         if(debug_nlo_diagrams) then
            write(logfile,*) "<diagram index='263'>"
         end if
         call ninja_diagram(numerator_diagram263, numerator_t3_diagram263, nume&
         &rator_t2_diagram263, &
          &  numerator_tmu_diagram263, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.NINJA_SUCCESS)
         if(debug_nlo_diagrams) then
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-finite' re='", +real(acc(0), ki), &
               & "' im='", aimag(acc(0)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-single' re='", +real(acc(-1), ki), &
               & "' im='", aimag(acc(-1)), "'/>"
            write(logfile,'(A30,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-double' re='", +real(acc(-2), ki), &
               & "' im='", aimag(acc(-2)), "'/>"
            write(logfile,'(A32,E24.16,A6,E24.16,A3)') &
               & "<result kind='nlo-rational' re='", +real(accr, ki), &
               & "' im='", aimag(accr), "'/>"
            if(ok) then
               write(logfile,'(A30)') "<flag name='ok' status='yes'/>"
            else
               write(logfile,'(A29)') "<flag name='ok' status='no'/>"
            end if
            write(logfile,*) "</diagram>"
         end if
         tot = tot  + acc
         totr = totr  + accr
      !------#] sum over reduction of single diagrams:
end subroutine ninja_reduce_group11
!-----#] subroutine ninja_reduce_group11:
!---#] reduce groups with ninja:
end module p2_gg_httbar_ninjah4

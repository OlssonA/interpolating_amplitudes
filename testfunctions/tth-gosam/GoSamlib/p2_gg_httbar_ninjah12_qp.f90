!
module     p2_gg_httbar_ninjah12_qp
   ! This file has been generated for ninja
   use quadninjago_module, only: ki_nin
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d85h12l1_qp, only: numerator_diagram85 => numerator_ninja
   use p2_gg_httbar_d85h12l121_qp, only: numerator_tmu_diagram85 => numerator_tmu
   use p2_gg_httbar_d85h12l131_qp, only: numerator_t3_diagram85 => numerator_t3
   use p2_gg_httbar_d85h12l132_qp, only: numerator_t2_diagram85 => numerator_t2
   use p2_gg_httbar_d262h12l1_qp, only: numerator_diagram262 => numerator_ninja
   use p2_gg_httbar_d262h12l121_qp, only: numerator_tmu_diagram262 => numerator_tmu
   use p2_gg_httbar_d262h12l131_qp, only: numerator_t3_diagram262 => numerator_t3
   use p2_gg_httbar_d262h12l132_qp, only: numerator_t2_diagram262 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram85, numerator_t3_diagram85, nu&
         &merator_t2_diagram85, &
          &  numerator_tmu_diagram85, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram262, numerator_t3_diagram262, &
         &numerator_t2_diagram262, &
          &  numerator_tmu_diagram262, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d91h12l1_qp, only: numerator_diagram91 => numerator_ninja
   use p2_gg_httbar_d91h12l121_qp, only: numerator_tmu_diagram91 => numerator_tmu
   use p2_gg_httbar_d91h12l131_qp, only: numerator_t3_diagram91 => numerator_t3
   use p2_gg_httbar_d91h12l132_qp, only: numerator_t2_diagram91 => numerator_t2
   use p2_gg_httbar_d260h12l1_qp, only: numerator_diagram260 => numerator_ninja
   use p2_gg_httbar_d260h12l121_qp, only: numerator_tmu_diagram260 => numerator_tmu
   use p2_gg_httbar_d260h12l131_qp, only: numerator_t3_diagram260 => numerator_t3
   use p2_gg_httbar_d260h12l132_qp, only: numerator_t2_diagram260 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram91, numerator_t3_diagram91, nu&
         &merator_t2_diagram91, &
          &  numerator_tmu_diagram91, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram260, numerator_t3_diagram260, &
         &numerator_t2_diagram260, &
          &  numerator_tmu_diagram260, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d29h12l1_qp, only: numerator_diagram29 => numerator_ninja
   use p2_gg_httbar_d29h12l121_qp, only: numerator_tmu_diagram29 => numerator_tmu
   use p2_gg_httbar_d29h12l131_qp, only: numerator_t3_diagram29 => numerator_t3
   use p2_gg_httbar_d29h12l132_qp, only: numerator_t2_diagram29 => numerator_t2
   use p2_gg_httbar_d31h12l1_qp, only: numerator_diagram31 => numerator_ninja
   use p2_gg_httbar_d31h12l121_qp, only: numerator_tmu_diagram31 => numerator_tmu
   use p2_gg_httbar_d31h12l131_qp, only: numerator_t3_diagram31 => numerator_t3
   use p2_gg_httbar_d31h12l132_qp, only: numerator_t2_diagram31 => numerator_t2
   use p2_gg_httbar_d33h12l1_qp, only: numerator_diagram33 => numerator_ninja
   use p2_gg_httbar_d33h12l121_qp, only: numerator_tmu_diagram33 => numerator_tmu
   use p2_gg_httbar_d33h12l131_qp, only: numerator_t3_diagram33 => numerator_t3
   use p2_gg_httbar_d33h12l132_qp, only: numerator_t2_diagram33 => numerator_t2
   use p2_gg_httbar_d74h12l1_qp, only: numerator_diagram74 => numerator_ninja
   use p2_gg_httbar_d74h12l121_qp, only: numerator_tmu_diagram74 => numerator_tmu
   use p2_gg_httbar_d74h12l131_qp, only: numerator_t3_diagram74 => numerator_t3
   use p2_gg_httbar_d74h12l132_qp, only: numerator_t2_diagram74 => numerator_t2
   use p2_gg_httbar_d77h12l1_qp, only: numerator_diagram77 => numerator_ninja
   use p2_gg_httbar_d77h12l121_qp, only: numerator_tmu_diagram77 => numerator_tmu
   use p2_gg_httbar_d77h12l131_qp, only: numerator_t3_diagram77 => numerator_t3
   use p2_gg_httbar_d77h12l132_qp, only: numerator_t2_diagram77 => numerator_t2
   use p2_gg_httbar_d82h12l1_qp, only: numerator_diagram82 => numerator_ninja
   use p2_gg_httbar_d82h12l121_qp, only: numerator_tmu_diagram82 => numerator_tmu
   use p2_gg_httbar_d82h12l131_qp, only: numerator_t3_diagram82 => numerator_t3
   use p2_gg_httbar_d82h12l132_qp, only: numerator_t2_diagram82 => numerator_t2
   use p2_gg_httbar_d90h12l1_qp, only: numerator_diagram90 => numerator_ninja
   use p2_gg_httbar_d90h12l121_qp, only: numerator_tmu_diagram90 => numerator_tmu
   use p2_gg_httbar_d90h12l131_qp, only: numerator_t3_diagram90 => numerator_t3
   use p2_gg_httbar_d90h12l132_qp, only: numerator_t2_diagram90 => numerator_t2
   use p2_gg_httbar_d148h12l1_qp, only: numerator_diagram148 => numerator_ninja
   use p2_gg_httbar_d148h12l121_qp, only: numerator_tmu_diagram148 => numerator_tmu
   use p2_gg_httbar_d148h12l131_qp, only: numerator_t3_diagram148 => numerator_t3
   use p2_gg_httbar_d148h12l132_qp, only: numerator_t2_diagram148 => numerator_t2
   use p2_gg_httbar_d163h12l1_qp, only: numerator_diagram163 => numerator_ninja
   use p2_gg_httbar_d163h12l121_qp, only: numerator_tmu_diagram163 => numerator_tmu
   use p2_gg_httbar_d163h12l131_qp, only: numerator_t3_diagram163 => numerator_t3
   use p2_gg_httbar_d163h12l132_qp, only: numerator_t2_diagram163 => numerator_t2
   use p2_gg_httbar_d258h12l1_qp, only: numerator_diagram258 => numerator_ninja
   use p2_gg_httbar_d258h12l121_qp, only: numerator_tmu_diagram258 => numerator_tmu
   use p2_gg_httbar_d258h12l131_qp, only: numerator_t3_diagram258 => numerator_t3
   use p2_gg_httbar_d258h12l132_qp, only: numerator_t2_diagram258 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram29, numerator_t3_diagram29, nu&
         &merator_t2_diagram29, &
          &  numerator_tmu_diagram29, &
          & 5, 3, 2, (/2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram31, numerator_t3_diagram31, nu&
         &merator_t2_diagram31, &
          &  numerator_tmu_diagram31, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram33, numerator_t3_diagram33, nu&
         &merator_t2_diagram33, &
          &  numerator_tmu_diagram33, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram74, numerator_t3_diagram74, nu&
         &merator_t2_diagram74, &
          &  numerator_tmu_diagram74, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram77, numerator_t3_diagram77, nu&
         &merator_t2_diagram77, &
          &  numerator_tmu_diagram77, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram82, numerator_t3_diagram82, nu&
         &merator_t2_diagram82, &
          &  numerator_tmu_diagram82, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram90, numerator_t3_diagram90, nu&
         &merator_t2_diagram90, &
          &  numerator_tmu_diagram90, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram148, numerator_t3_diagram148, &
         &numerator_t2_diagram148, &
          &  numerator_tmu_diagram148, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram163, numerator_t3_diagram163, &
         &numerator_t2_diagram163, &
          &  numerator_tmu_diagram163, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram258, numerator_t3_diagram258, &
         &numerator_t2_diagram258, &
          &  numerator_tmu_diagram258, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d1h12l1_qp, only: numerator_diagram1 => numerator_ninja
   use p2_gg_httbar_d1h12l121_qp, only: numerator_tmu_diagram1 => numerator_tmu
   use p2_gg_httbar_d1h12l131_qp, only: numerator_t3_diagram1 => numerator_t3
   use p2_gg_httbar_d1h12l132_qp, only: numerator_t2_diagram1 => numerator_t2
   use p2_gg_httbar_d2h12l1_qp, only: numerator_diagram2 => numerator_ninja
   use p2_gg_httbar_d2h12l121_qp, only: numerator_tmu_diagram2 => numerator_tmu
   use p2_gg_httbar_d2h12l131_qp, only: numerator_t3_diagram2 => numerator_t3
   use p2_gg_httbar_d2h12l132_qp, only: numerator_t2_diagram2 => numerator_t2
   use p2_gg_httbar_d3h12l1_qp, only: numerator_diagram3 => numerator_ninja
   use p2_gg_httbar_d3h12l121_qp, only: numerator_tmu_diagram3 => numerator_tmu
   use p2_gg_httbar_d3h12l131_qp, only: numerator_t3_diagram3 => numerator_t3
   use p2_gg_httbar_d3h12l132_qp, only: numerator_t2_diagram3 => numerator_t2
   use p2_gg_httbar_d5h12l1_qp, only: numerator_diagram5 => numerator_ninja
   use p2_gg_httbar_d5h12l121_qp, only: numerator_tmu_diagram5 => numerator_tmu
   use p2_gg_httbar_d5h12l131_qp, only: numerator_t3_diagram5 => numerator_t3
   use p2_gg_httbar_d5h12l132_qp, only: numerator_t2_diagram5 => numerator_t2
   use p2_gg_httbar_d6h12l1_qp, only: numerator_diagram6 => numerator_ninja
   use p2_gg_httbar_d6h12l121_qp, only: numerator_tmu_diagram6 => numerator_tmu
   use p2_gg_httbar_d6h12l131_qp, only: numerator_t3_diagram6 => numerator_t3
   use p2_gg_httbar_d6h12l132_qp, only: numerator_t2_diagram6 => numerator_t2
   use p2_gg_httbar_d7h12l1_qp, only: numerator_diagram7 => numerator_ninja
   use p2_gg_httbar_d7h12l121_qp, only: numerator_tmu_diagram7 => numerator_tmu
   use p2_gg_httbar_d7h12l131_qp, only: numerator_t3_diagram7 => numerator_t3
   use p2_gg_httbar_d7h12l132_qp, only: numerator_t2_diagram7 => numerator_t2
   use p2_gg_httbar_d11h12l1_qp, only: numerator_diagram11 => numerator_ninja
   use p2_gg_httbar_d11h12l121_qp, only: numerator_tmu_diagram11 => numerator_tmu
   use p2_gg_httbar_d11h12l131_qp, only: numerator_t3_diagram11 => numerator_t3
   use p2_gg_httbar_d11h12l132_qp, only: numerator_t2_diagram11 => numerator_t2
   use p2_gg_httbar_d13h12l1_qp, only: numerator_diagram13 => numerator_ninja
   use p2_gg_httbar_d13h12l121_qp, only: numerator_tmu_diagram13 => numerator_tmu
   use p2_gg_httbar_d13h12l131_qp, only: numerator_t3_diagram13 => numerator_t3
   use p2_gg_httbar_d13h12l132_qp, only: numerator_t2_diagram13 => numerator_t2
   use p2_gg_httbar_d26h12l1_qp, only: numerator_diagram26 => numerator_ninja
   use p2_gg_httbar_d26h12l121_qp, only: numerator_tmu_diagram26 => numerator_tmu
   use p2_gg_httbar_d26h12l131_qp, only: numerator_t3_diagram26 => numerator_t3
   use p2_gg_httbar_d26h12l132_qp, only: numerator_t2_diagram26 => numerator_t2
   use p2_gg_httbar_d28h12l1_qp, only: numerator_diagram28 => numerator_ninja
   use p2_gg_httbar_d28h12l121_qp, only: numerator_tmu_diagram28 => numerator_tmu
   use p2_gg_httbar_d28h12l131_qp, only: numerator_t3_diagram28 => numerator_t3
   use p2_gg_httbar_d28h12l132_qp, only: numerator_t2_diagram28 => numerator_t2
   use p2_gg_httbar_d35h12l1_qp, only: numerator_diagram35 => numerator_ninja
   use p2_gg_httbar_d35h12l121_qp, only: numerator_tmu_diagram35 => numerator_tmu
   use p2_gg_httbar_d35h12l131_qp, only: numerator_t3_diagram35 => numerator_t3
   use p2_gg_httbar_d35h12l132_qp, only: numerator_t2_diagram35 => numerator_t2
   use p2_gg_httbar_d46h12l1_qp, only: numerator_diagram46 => numerator_ninja
   use p2_gg_httbar_d46h12l121_qp, only: numerator_tmu_diagram46 => numerator_tmu
   use p2_gg_httbar_d46h12l131_qp, only: numerator_t3_diagram46 => numerator_t3
   use p2_gg_httbar_d46h12l132_qp, only: numerator_t2_diagram46 => numerator_t2
   use p2_gg_httbar_d66h12l1_qp, only: numerator_diagram66 => numerator_ninja
   use p2_gg_httbar_d66h12l121_qp, only: numerator_tmu_diagram66 => numerator_tmu
   use p2_gg_httbar_d66h12l131_qp, only: numerator_t3_diagram66 => numerator_t3
   use p2_gg_httbar_d66h12l132_qp, only: numerator_t2_diagram66 => numerator_t2
   use p2_gg_httbar_d71h12l1_qp, only: numerator_diagram71 => numerator_ninja
   use p2_gg_httbar_d71h12l121_qp, only: numerator_tmu_diagram71 => numerator_tmu
   use p2_gg_httbar_d71h12l131_qp, only: numerator_t3_diagram71 => numerator_t3
   use p2_gg_httbar_d71h12l132_qp, only: numerator_t2_diagram71 => numerator_t2
   use p2_gg_httbar_d80h12l1_qp, only: numerator_diagram80 => numerator_ninja
   use p2_gg_httbar_d80h12l121_qp, only: numerator_tmu_diagram80 => numerator_tmu
   use p2_gg_httbar_d80h12l131_qp, only: numerator_t3_diagram80 => numerator_t3
   use p2_gg_httbar_d80h12l132_qp, only: numerator_t2_diagram80 => numerator_t2
   use p2_gg_httbar_d84h12l1_qp, only: numerator_diagram84 => numerator_ninja
   use p2_gg_httbar_d84h12l121_qp, only: numerator_tmu_diagram84 => numerator_tmu
   use p2_gg_httbar_d84h12l131_qp, only: numerator_t3_diagram84 => numerator_t3
   use p2_gg_httbar_d84h12l132_qp, only: numerator_t2_diagram84 => numerator_t2
   use p2_gg_httbar_d88h12l1_qp, only: numerator_diagram88 => numerator_ninja
   use p2_gg_httbar_d88h12l121_qp, only: numerator_tmu_diagram88 => numerator_tmu
   use p2_gg_httbar_d88h12l131_qp, only: numerator_t3_diagram88 => numerator_t3
   use p2_gg_httbar_d88h12l132_qp, only: numerator_t2_diagram88 => numerator_t2
   use p2_gg_httbar_d133h12l1_qp, only: numerator_diagram133 => numerator_ninja
   use p2_gg_httbar_d133h12l121_qp, only: numerator_tmu_diagram133 => numerator_tmu
   use p2_gg_httbar_d133h12l131_qp, only: numerator_t3_diagram133 => numerator_t3
   use p2_gg_httbar_d133h12l132_qp, only: numerator_t2_diagram133 => numerator_t2
   use p2_gg_httbar_d178h12l1_qp, only: numerator_diagram178 => numerator_ninja
   use p2_gg_httbar_d178h12l121_qp, only: numerator_tmu_diagram178 => numerator_tmu
   use p2_gg_httbar_d178h12l131_qp, only: numerator_t3_diagram178 => numerator_t3
   use p2_gg_httbar_d178h12l132_qp, only: numerator_t2_diagram178 => numerator_t2
   use p2_gg_httbar_d195h12l1_qp, only: numerator_diagram195 => numerator_ninja
   use p2_gg_httbar_d195h12l121_qp, only: numerator_tmu_diagram195 => numerator_tmu
   use p2_gg_httbar_d195h12l131_qp, only: numerator_t3_diagram195 => numerator_t3
   use p2_gg_httbar_d195h12l132_qp, only: numerator_t2_diagram195 => numerator_t2
   use p2_gg_httbar_d256h12l1_qp, only: numerator_diagram256 => numerator_ninja
   use p2_gg_httbar_d256h12l121_qp, only: numerator_tmu_diagram256 => numerator_tmu
   use p2_gg_httbar_d256h12l131_qp, only: numerator_t3_diagram256 => numerator_t3
   use p2_gg_httbar_d256h12l132_qp, only: numerator_t2_diagram256 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram1, numerator_t3_diagram1, nume&
         &rator_t2_diagram1, &
          &  numerator_tmu_diagram1, &
          & 5, 3, 1, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram2, numerator_t3_diagram2, nume&
         &rator_t2_diagram2, &
          &  numerator_tmu_diagram2, &
          & 5, 3, 1, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram3, numerator_t3_diagram3, nume&
         &rator_t2_diagram3, &
          &  numerator_tmu_diagram3, &
          & 5, 2, 1, (/1,4/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram5, numerator_t3_diagram5, nume&
         &rator_t2_diagram5, &
          &  numerator_tmu_diagram5, &
          & 5, 4, 2, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram6, numerator_t3_diagram6, nume&
         &rator_t2_diagram6, &
          &  numerator_tmu_diagram6, &
          & 5, 2, 1, (/1,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram7, numerator_t3_diagram7, nume&
         &rator_t2_diagram7, &
          &  numerator_tmu_diagram7, &
          & 5, 2, 1, (/4,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram11, numerator_t3_diagram11, nu&
         &merator_t2_diagram11, &
          &  numerator_tmu_diagram11, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram13, numerator_t3_diagram13, nu&
         &merator_t2_diagram13, &
          &  numerator_tmu_diagram13, &
          & 5, 3, 2, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram26, numerator_t3_diagram26, nu&
         &merator_t2_diagram26, &
          &  numerator_tmu_diagram26, &
          & 5, 3, 2, (/2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram28, numerator_t3_diagram28, nu&
         &merator_t2_diagram28, &
          &  numerator_tmu_diagram28, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram35, numerator_t3_diagram35, nu&
         &merator_t2_diagram35, &
          &  numerator_tmu_diagram35, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram46, numerator_t3_diagram46, nu&
         &merator_t2_diagram46, &
          &  numerator_tmu_diagram46, &
          & 5, 2, 2, (/1,4/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram66, numerator_t3_diagram66, nu&
         &merator_t2_diagram66, &
          &  numerator_tmu_diagram66, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram71, numerator_t3_diagram71, nu&
         &merator_t2_diagram71, &
          &  numerator_tmu_diagram71, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram80, numerator_t3_diagram80, nu&
         &merator_t2_diagram80, &
          &  numerator_tmu_diagram80, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram84, numerator_t3_diagram84, nu&
         &merator_t2_diagram84, &
          &  numerator_tmu_diagram84, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram88, numerator_t3_diagram88, nu&
         &merator_t2_diagram88, &
          &  numerator_tmu_diagram88, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram133, numerator_t3_diagram133, &
         &numerator_t2_diagram133, &
          &  numerator_tmu_diagram133, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram178, numerator_t3_diagram178, &
         &numerator_t2_diagram178, &
          &  numerator_tmu_diagram178, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram195, numerator_t3_diagram195, &
         &numerator_t2_diagram195, &
          &  numerator_tmu_diagram195, &
          & 5, 3, 3, (/1,4,5/), &
          & Vi, s_mat, scale2, istop0, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram256, numerator_t3_diagram256, &
         &numerator_t2_diagram256, &
          &  numerator_tmu_diagram256, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d10h12l1_qp, only: numerator_diagram10 => numerator_ninja
   use p2_gg_httbar_d10h12l121_qp, only: numerator_tmu_diagram10 => numerator_tmu
   use p2_gg_httbar_d10h12l131_qp, only: numerator_t3_diagram10 => numerator_t3
   use p2_gg_httbar_d10h12l132_qp, only: numerator_t2_diagram10 => numerator_t2
   use p2_gg_httbar_d34h12l1_qp, only: numerator_diagram34 => numerator_ninja
   use p2_gg_httbar_d34h12l121_qp, only: numerator_tmu_diagram34 => numerator_tmu
   use p2_gg_httbar_d34h12l131_qp, only: numerator_t3_diagram34 => numerator_t3
   use p2_gg_httbar_d34h12l132_qp, only: numerator_t2_diagram34 => numerator_t2
   use p2_gg_httbar_d36h12l1_qp, only: numerator_diagram36 => numerator_ninja
   use p2_gg_httbar_d36h12l121_qp, only: numerator_tmu_diagram36 => numerator_tmu
   use p2_gg_httbar_d36h12l131_qp, only: numerator_t3_diagram36 => numerator_t3
   use p2_gg_httbar_d36h12l132_qp, only: numerator_t2_diagram36 => numerator_t2
   use p2_gg_httbar_d38h12l1_qp, only: numerator_diagram38 => numerator_ninja
   use p2_gg_httbar_d38h12l121_qp, only: numerator_tmu_diagram38 => numerator_tmu
   use p2_gg_httbar_d38h12l131_qp, only: numerator_t3_diagram38 => numerator_t3
   use p2_gg_httbar_d38h12l132_qp, only: numerator_t2_diagram38 => numerator_t2
   use p2_gg_httbar_d67h12l1_qp, only: numerator_diagram67 => numerator_ninja
   use p2_gg_httbar_d67h12l121_qp, only: numerator_tmu_diagram67 => numerator_tmu
   use p2_gg_httbar_d67h12l131_qp, only: numerator_t3_diagram67 => numerator_t3
   use p2_gg_httbar_d67h12l132_qp, only: numerator_t2_diagram67 => numerator_t2
   use p2_gg_httbar_d79h12l1_qp, only: numerator_diagram79 => numerator_ninja
   use p2_gg_httbar_d79h12l121_qp, only: numerator_tmu_diagram79 => numerator_tmu
   use p2_gg_httbar_d79h12l131_qp, only: numerator_t3_diagram79 => numerator_t3
   use p2_gg_httbar_d79h12l132_qp, only: numerator_t2_diagram79 => numerator_t2
   use p2_gg_httbar_d83h12l1_qp, only: numerator_diagram83 => numerator_ninja
   use p2_gg_httbar_d83h12l121_qp, only: numerator_tmu_diagram83 => numerator_tmu
   use p2_gg_httbar_d83h12l131_qp, only: numerator_t3_diagram83 => numerator_t3
   use p2_gg_httbar_d83h12l132_qp, only: numerator_t2_diagram83 => numerator_t2
   use p2_gg_httbar_d130h12l1_qp, only: numerator_diagram130 => numerator_ninja
   use p2_gg_httbar_d130h12l121_qp, only: numerator_tmu_diagram130 => numerator_tmu
   use p2_gg_httbar_d130h12l131_qp, only: numerator_t3_diagram130 => numerator_t3
   use p2_gg_httbar_d130h12l132_qp, only: numerator_t2_diagram130 => numerator_t2
   use p2_gg_httbar_d132h12l1_qp, only: numerator_diagram132 => numerator_ninja
   use p2_gg_httbar_d132h12l121_qp, only: numerator_tmu_diagram132 => numerator_tmu
   use p2_gg_httbar_d132h12l131_qp, only: numerator_t3_diagram132 => numerator_t3
   use p2_gg_httbar_d132h12l132_qp, only: numerator_t2_diagram132 => numerator_t2
   use p2_gg_httbar_d254h12l1_qp, only: numerator_diagram254 => numerator_ninja
   use p2_gg_httbar_d254h12l121_qp, only: numerator_tmu_diagram254 => numerator_tmu
   use p2_gg_httbar_d254h12l131_qp, only: numerator_t3_diagram254 => numerator_t3
   use p2_gg_httbar_d254h12l132_qp, only: numerator_t2_diagram254 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram10, numerator_t3_diagram10, nu&
         &merator_t2_diagram10, &
          &  numerator_tmu_diagram10, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram34, numerator_t3_diagram34, nu&
         &merator_t2_diagram34, &
          &  numerator_tmu_diagram34, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram36, numerator_t3_diagram36, nu&
         &merator_t2_diagram36, &
          &  numerator_tmu_diagram36, &
          & 5, 2, 2, (/1,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram38, numerator_t3_diagram38, nu&
         &merator_t2_diagram38, &
          &  numerator_tmu_diagram38, &
          & 5, 2, 2, (/3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram67, numerator_t3_diagram67, nu&
         &merator_t2_diagram67, &
          &  numerator_tmu_diagram67, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram79, numerator_t3_diagram79, nu&
         &merator_t2_diagram79, &
          &  numerator_tmu_diagram79, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram83, numerator_t3_diagram83, nu&
         &merator_t2_diagram83, &
          &  numerator_tmu_diagram83, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram130, numerator_t3_diagram130, &
         &numerator_t2_diagram130, &
          &  numerator_tmu_diagram130, &
          & 5, 3, 2, (/1,2,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram132, numerator_t3_diagram132, &
         &numerator_t2_diagram132, &
          &  numerator_tmu_diagram132, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram254, numerator_t3_diagram254, &
         &numerator_t2_diagram254, &
          &  numerator_tmu_diagram254, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d69h12l1_qp, only: numerator_diagram69 => numerator_ninja
   use p2_gg_httbar_d69h12l121_qp, only: numerator_tmu_diagram69 => numerator_tmu
   use p2_gg_httbar_d69h12l131_qp, only: numerator_t3_diagram69 => numerator_t3
   use p2_gg_httbar_d69h12l132_qp, only: numerator_t2_diagram69 => numerator_t2
   use p2_gg_httbar_d78h12l1_qp, only: numerator_diagram78 => numerator_ninja
   use p2_gg_httbar_d78h12l121_qp, only: numerator_tmu_diagram78 => numerator_tmu
   use p2_gg_httbar_d78h12l131_qp, only: numerator_t3_diagram78 => numerator_t3
   use p2_gg_httbar_d78h12l132_qp, only: numerator_t2_diagram78 => numerator_t2
   use p2_gg_httbar_d125h12l1_qp, only: numerator_diagram125 => numerator_ninja
   use p2_gg_httbar_d125h12l121_qp, only: numerator_tmu_diagram125 => numerator_tmu
   use p2_gg_httbar_d125h12l131_qp, only: numerator_t3_diagram125 => numerator_t3
   use p2_gg_httbar_d125h12l132_qp, only: numerator_t2_diagram125 => numerator_t2
   use p2_gg_httbar_d259h12l1_qp, only: numerator_diagram259 => numerator_ninja
   use p2_gg_httbar_d259h12l121_qp, only: numerator_tmu_diagram259 => numerator_tmu
   use p2_gg_httbar_d259h12l131_qp, only: numerator_t3_diagram259 => numerator_t3
   use p2_gg_httbar_d259h12l132_qp, only: numerator_t2_diagram259 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram69, numerator_t3_diagram69, nu&
         &merator_t2_diagram69, &
          &  numerator_tmu_diagram69, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram78, numerator_t3_diagram78, nu&
         &merator_t2_diagram78, &
          &  numerator_tmu_diagram78, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram125, numerator_t3_diagram125, &
         &numerator_t2_diagram125, &
          &  numerator_tmu_diagram125, &
          & 5, 4, 4, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram259, numerator_t3_diagram259, &
         &numerator_t2_diagram259, &
          &  numerator_tmu_diagram259, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d30h12l1_qp, only: numerator_diagram30 => numerator_ninja
   use p2_gg_httbar_d30h12l121_qp, only: numerator_tmu_diagram30 => numerator_tmu
   use p2_gg_httbar_d30h12l131_qp, only: numerator_t3_diagram30 => numerator_t3
   use p2_gg_httbar_d30h12l132_qp, only: numerator_t2_diagram30 => numerator_t2
   use p2_gg_httbar_d42h12l1_qp, only: numerator_diagram42 => numerator_ninja
   use p2_gg_httbar_d42h12l121_qp, only: numerator_tmu_diagram42 => numerator_tmu
   use p2_gg_httbar_d42h12l131_qp, only: numerator_t3_diagram42 => numerator_t3
   use p2_gg_httbar_d42h12l132_qp, only: numerator_t2_diagram42 => numerator_t2
   use p2_gg_httbar_d73h12l1_qp, only: numerator_diagram73 => numerator_ninja
   use p2_gg_httbar_d73h12l121_qp, only: numerator_tmu_diagram73 => numerator_tmu
   use p2_gg_httbar_d73h12l131_qp, only: numerator_t3_diagram73 => numerator_t3
   use p2_gg_httbar_d73h12l132_qp, only: numerator_t2_diagram73 => numerator_t2
   use p2_gg_httbar_d81h12l1_qp, only: numerator_diagram81 => numerator_ninja
   use p2_gg_httbar_d81h12l121_qp, only: numerator_tmu_diagram81 => numerator_tmu
   use p2_gg_httbar_d81h12l131_qp, only: numerator_t3_diagram81 => numerator_t3
   use p2_gg_httbar_d81h12l132_qp, only: numerator_t2_diagram81 => numerator_t2
   use p2_gg_httbar_d162h12l1_qp, only: numerator_diagram162 => numerator_ninja
   use p2_gg_httbar_d162h12l121_qp, only: numerator_tmu_diagram162 => numerator_tmu
   use p2_gg_httbar_d162h12l131_qp, only: numerator_t3_diagram162 => numerator_t3
   use p2_gg_httbar_d162h12l132_qp, only: numerator_t2_diagram162 => numerator_t2
   use p2_gg_httbar_d257h12l1_qp, only: numerator_diagram257 => numerator_ninja
   use p2_gg_httbar_d257h12l121_qp, only: numerator_tmu_diagram257 => numerator_tmu
   use p2_gg_httbar_d257h12l131_qp, only: numerator_t3_diagram257 => numerator_t3
   use p2_gg_httbar_d257h12l132_qp, only: numerator_t2_diagram257 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram30, numerator_t3_diagram30, nu&
         &merator_t2_diagram30, &
          &  numerator_tmu_diagram30, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram42, numerator_t3_diagram42, nu&
         &merator_t2_diagram42, &
          &  numerator_tmu_diagram42, &
          & 5, 2, 2, (/2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram73, numerator_t3_diagram73, nu&
         &merator_t2_diagram73, &
          &  numerator_tmu_diagram73, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram81, numerator_t3_diagram81, nu&
         &merator_t2_diagram81, &
          &  numerator_tmu_diagram81, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram162, numerator_t3_diagram162, &
         &numerator_t2_diagram162, &
          &  numerator_tmu_diagram162, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram257, numerator_t3_diagram257, &
         &numerator_t2_diagram257, &
          &  numerator_tmu_diagram257, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d12h12l1_qp, only: numerator_diagram12 => numerator_ninja
   use p2_gg_httbar_d12h12l121_qp, only: numerator_tmu_diagram12 => numerator_tmu
   use p2_gg_httbar_d12h12l131_qp, only: numerator_t3_diagram12 => numerator_t3
   use p2_gg_httbar_d12h12l132_qp, only: numerator_t2_diagram12 => numerator_t2
   use p2_gg_httbar_d22h12l1_qp, only: numerator_diagram22 => numerator_ninja
   use p2_gg_httbar_d22h12l121_qp, only: numerator_tmu_diagram22 => numerator_tmu
   use p2_gg_httbar_d22h12l131_qp, only: numerator_t3_diagram22 => numerator_t3
   use p2_gg_httbar_d22h12l132_qp, only: numerator_t2_diagram22 => numerator_t2
   use p2_gg_httbar_d32h12l1_qp, only: numerator_diagram32 => numerator_ninja
   use p2_gg_httbar_d32h12l121_qp, only: numerator_tmu_diagram32 => numerator_tmu
   use p2_gg_httbar_d32h12l131_qp, only: numerator_t3_diagram32 => numerator_t3
   use p2_gg_httbar_d32h12l132_qp, only: numerator_t2_diagram32 => numerator_t2
   use p2_gg_httbar_d37h12l1_qp, only: numerator_diagram37 => numerator_ninja
   use p2_gg_httbar_d37h12l121_qp, only: numerator_tmu_diagram37 => numerator_tmu
   use p2_gg_httbar_d37h12l131_qp, only: numerator_t3_diagram37 => numerator_t3
   use p2_gg_httbar_d37h12l132_qp, only: numerator_t2_diagram37 => numerator_t2
   use p2_gg_httbar_d40h12l1_qp, only: numerator_diagram40 => numerator_ninja
   use p2_gg_httbar_d40h12l121_qp, only: numerator_tmu_diagram40 => numerator_tmu
   use p2_gg_httbar_d40h12l131_qp, only: numerator_t3_diagram40 => numerator_t3
   use p2_gg_httbar_d40h12l132_qp, only: numerator_t2_diagram40 => numerator_t2
   use p2_gg_httbar_d50h12l1_qp, only: numerator_diagram50 => numerator_ninja
   use p2_gg_httbar_d50h12l121_qp, only: numerator_tmu_diagram50 => numerator_tmu
   use p2_gg_httbar_d50h12l131_qp, only: numerator_t3_diagram50 => numerator_t3
   use p2_gg_httbar_d50h12l132_qp, only: numerator_t2_diagram50 => numerator_t2
   use p2_gg_httbar_d68h12l1_qp, only: numerator_diagram68 => numerator_ninja
   use p2_gg_httbar_d68h12l121_qp, only: numerator_tmu_diagram68 => numerator_tmu
   use p2_gg_httbar_d68h12l131_qp, only: numerator_t3_diagram68 => numerator_t3
   use p2_gg_httbar_d68h12l132_qp, only: numerator_t2_diagram68 => numerator_t2
   use p2_gg_httbar_d76h12l1_qp, only: numerator_diagram76 => numerator_ninja
   use p2_gg_httbar_d76h12l121_qp, only: numerator_tmu_diagram76 => numerator_tmu
   use p2_gg_httbar_d76h12l131_qp, only: numerator_t3_diagram76 => numerator_t3
   use p2_gg_httbar_d76h12l132_qp, only: numerator_t2_diagram76 => numerator_t2
   use p2_gg_httbar_d89h12l1_qp, only: numerator_diagram89 => numerator_ninja
   use p2_gg_httbar_d89h12l121_qp, only: numerator_tmu_diagram89 => numerator_tmu
   use p2_gg_httbar_d89h12l131_qp, only: numerator_t3_diagram89 => numerator_t3
   use p2_gg_httbar_d89h12l132_qp, only: numerator_t2_diagram89 => numerator_t2
   use p2_gg_httbar_d101h12l1_qp, only: numerator_diagram101 => numerator_ninja
   use p2_gg_httbar_d101h12l121_qp, only: numerator_tmu_diagram101 => numerator_tmu
   use p2_gg_httbar_d101h12l131_qp, only: numerator_t3_diagram101 => numerator_t3
   use p2_gg_httbar_d101h12l132_qp, only: numerator_t2_diagram101 => numerator_t2
   use p2_gg_httbar_d129h12l1_qp, only: numerator_diagram129 => numerator_ninja
   use p2_gg_httbar_d129h12l121_qp, only: numerator_tmu_diagram129 => numerator_tmu
   use p2_gg_httbar_d129h12l131_qp, only: numerator_t3_diagram129 => numerator_t3
   use p2_gg_httbar_d129h12l132_qp, only: numerator_t2_diagram129 => numerator_t2
   use p2_gg_httbar_d147h12l1_qp, only: numerator_diagram147 => numerator_ninja
   use p2_gg_httbar_d147h12l121_qp, only: numerator_tmu_diagram147 => numerator_tmu
   use p2_gg_httbar_d147h12l131_qp, only: numerator_t3_diagram147 => numerator_t3
   use p2_gg_httbar_d147h12l132_qp, only: numerator_t2_diagram147 => numerator_t2
   use p2_gg_httbar_d172h12l1_qp, only: numerator_diagram172 => numerator_ninja
   use p2_gg_httbar_d172h12l121_qp, only: numerator_tmu_diagram172 => numerator_tmu
   use p2_gg_httbar_d172h12l131_qp, only: numerator_t3_diagram172 => numerator_t3
   use p2_gg_httbar_d172h12l132_qp, only: numerator_t2_diagram172 => numerator_t2
   use p2_gg_httbar_d203h12l1_qp, only: numerator_diagram203 => numerator_ninja
   use p2_gg_httbar_d203h12l121_qp, only: numerator_tmu_diagram203 => numerator_tmu
   use p2_gg_httbar_d203h12l131_qp, only: numerator_t3_diagram203 => numerator_t3
   use p2_gg_httbar_d203h12l132_qp, only: numerator_t2_diagram203 => numerator_t2
   use p2_gg_httbar_d253h12l1_qp, only: numerator_diagram253 => numerator_ninja
   use p2_gg_httbar_d253h12l121_qp, only: numerator_tmu_diagram253 => numerator_tmu
   use p2_gg_httbar_d253h12l131_qp, only: numerator_t3_diagram253 => numerator_t3
   use p2_gg_httbar_d253h12l132_qp, only: numerator_t2_diagram253 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram12, numerator_t3_diagram12, nu&
         &merator_t2_diagram12, &
          &  numerator_tmu_diagram12, &
          & 5, 3, 2, (/1,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram22, numerator_t3_diagram22, nu&
         &merator_t2_diagram22, &
          &  numerator_tmu_diagram22, &
          & 5, 3, 3, (/1,2,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram32, numerator_t3_diagram32, nu&
         &merator_t2_diagram32, &
          &  numerator_tmu_diagram32, &
          & 5, 3, 2, (/1,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram37, numerator_t3_diagram37, nu&
         &merator_t2_diagram37, &
          &  numerator_tmu_diagram37, &
          & 5, 2, 2, (/1,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram40, numerator_t3_diagram40, nu&
         &merator_t2_diagram40, &
          &  numerator_tmu_diagram40, &
          & 5, 2, 2, (/3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram50, numerator_t3_diagram50, nu&
         &merator_t2_diagram50, &
          &  numerator_tmu_diagram50, &
          & 5, 2, 2, (/1,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram68, numerator_t3_diagram68, nu&
         &merator_t2_diagram68, &
          &  numerator_tmu_diagram68, &
          & 5, 4, 3, (/1,2,3,4/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram76, numerator_t3_diagram76, nu&
         &merator_t2_diagram76, &
          &  numerator_tmu_diagram76, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram89, numerator_t3_diagram89, nu&
         &merator_t2_diagram89, &
          &  numerator_tmu_diagram89, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram101, numerator_t3_diagram101, &
         &numerator_t2_diagram101, &
          &  numerator_tmu_diagram101, &
          & 5, 4, 4, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram129, numerator_t3_diagram129, &
         &numerator_t2_diagram129, &
          &  numerator_tmu_diagram129, &
          & 5, 3, 2, (/1,2,3/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram147, numerator_t3_diagram147, &
         &numerator_t2_diagram147, &
          &  numerator_tmu_diagram147, &
          & 5, 3, 2, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram172, numerator_t3_diagram172, &
         &numerator_t2_diagram172, &
          &  numerator_tmu_diagram172, &
          & 5, 3, 3, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram203, numerator_t3_diagram203, &
         &numerator_t2_diagram203, &
          &  numerator_tmu_diagram203, &
          & 5, 3, 3, (/1,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram253, numerator_t3_diagram253, &
         &numerator_t2_diagram253, &
          &  numerator_tmu_diagram253, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d27h12l1_qp, only: numerator_diagram27 => numerator_ninja
   use p2_gg_httbar_d27h12l121_qp, only: numerator_tmu_diagram27 => numerator_tmu
   use p2_gg_httbar_d27h12l131_qp, only: numerator_t3_diagram27 => numerator_t3
   use p2_gg_httbar_d27h12l132_qp, only: numerator_t2_diagram27 => numerator_t2
   use p2_gg_httbar_d44h12l1_qp, only: numerator_diagram44 => numerator_ninja
   use p2_gg_httbar_d44h12l121_qp, only: numerator_tmu_diagram44 => numerator_tmu
   use p2_gg_httbar_d44h12l131_qp, only: numerator_t3_diagram44 => numerator_t3
   use p2_gg_httbar_d44h12l132_qp, only: numerator_t2_diagram44 => numerator_t2
   use p2_gg_httbar_d70h12l1_qp, only: numerator_diagram70 => numerator_ninja
   use p2_gg_httbar_d70h12l121_qp, only: numerator_tmu_diagram70 => numerator_tmu
   use p2_gg_httbar_d70h12l131_qp, only: numerator_t3_diagram70 => numerator_t3
   use p2_gg_httbar_d70h12l132_qp, only: numerator_t2_diagram70 => numerator_t2
   use p2_gg_httbar_d87h12l1_qp, only: numerator_diagram87 => numerator_ninja
   use p2_gg_httbar_d87h12l121_qp, only: numerator_tmu_diagram87 => numerator_tmu
   use p2_gg_httbar_d87h12l131_qp, only: numerator_t3_diagram87 => numerator_t3
   use p2_gg_httbar_d87h12l132_qp, only: numerator_t2_diagram87 => numerator_t2
   use p2_gg_httbar_d113h12l1_qp, only: numerator_diagram113 => numerator_ninja
   use p2_gg_httbar_d113h12l121_qp, only: numerator_tmu_diagram113 => numerator_tmu
   use p2_gg_httbar_d113h12l131_qp, only: numerator_t3_diagram113 => numerator_t3
   use p2_gg_httbar_d113h12l132_qp, only: numerator_t2_diagram113 => numerator_t2
   use p2_gg_httbar_d142h12l1_qp, only: numerator_diagram142 => numerator_ninja
   use p2_gg_httbar_d142h12l121_qp, only: numerator_tmu_diagram142 => numerator_tmu
   use p2_gg_httbar_d142h12l131_qp, only: numerator_t3_diagram142 => numerator_t3
   use p2_gg_httbar_d142h12l132_qp, only: numerator_t2_diagram142 => numerator_t2
   use p2_gg_httbar_d177h12l1_qp, only: numerator_diagram177 => numerator_ninja
   use p2_gg_httbar_d177h12l121_qp, only: numerator_tmu_diagram177 => numerator_tmu
   use p2_gg_httbar_d177h12l131_qp, only: numerator_t3_diagram177 => numerator_t3
   use p2_gg_httbar_d177h12l132_qp, only: numerator_t2_diagram177 => numerator_t2
   use p2_gg_httbar_d255h12l1_qp, only: numerator_diagram255 => numerator_ninja
   use p2_gg_httbar_d255h12l121_qp, only: numerator_tmu_diagram255 => numerator_tmu
   use p2_gg_httbar_d255h12l131_qp, only: numerator_t3_diagram255 => numerator_t3
   use p2_gg_httbar_d255h12l132_qp, only: numerator_t2_diagram255 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram27, numerator_t3_diagram27, nu&
         &merator_t2_diagram27, &
          &  numerator_tmu_diagram27, &
          & 5, 3, 2, (/2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram44, numerator_t3_diagram44, nu&
         &merator_t2_diagram44, &
          &  numerator_tmu_diagram44, &
          & 5, 2, 2, (/2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram70, numerator_t3_diagram70, nu&
         &merator_t2_diagram70, &
          &  numerator_tmu_diagram70, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram87, numerator_t3_diagram87, nu&
         &merator_t2_diagram87, &
          &  numerator_tmu_diagram87, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram113, numerator_t3_diagram113, &
         &numerator_t2_diagram113, &
          &  numerator_tmu_diagram113, &
          & 5, 4, 4, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram142, numerator_t3_diagram142, &
         &numerator_t2_diagram142, &
          &  numerator_tmu_diagram142, &
          & 5, 3, 3, (/3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram177, numerator_t3_diagram177, &
         &numerator_t2_diagram177, &
          &  numerator_tmu_diagram177, &
          & 5, 3, 2, (/1,2,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram255, numerator_t3_diagram255, &
         &numerator_t2_diagram255, &
          &  numerator_tmu_diagram255, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d72h12l1_qp, only: numerator_diagram72 => numerator_ninja
   use p2_gg_httbar_d72h12l121_qp, only: numerator_tmu_diagram72 => numerator_tmu
   use p2_gg_httbar_d72h12l131_qp, only: numerator_t3_diagram72 => numerator_t3
   use p2_gg_httbar_d72h12l132_qp, only: numerator_t2_diagram72 => numerator_t2
   use p2_gg_httbar_d75h12l1_qp, only: numerator_diagram75 => numerator_ninja
   use p2_gg_httbar_d75h12l121_qp, only: numerator_tmu_diagram75 => numerator_tmu
   use p2_gg_httbar_d75h12l131_qp, only: numerator_t3_diagram75 => numerator_t3
   use p2_gg_httbar_d75h12l132_qp, only: numerator_t2_diagram75 => numerator_t2
   use p2_gg_httbar_d261h12l1_qp, only: numerator_diagram261 => numerator_ninja
   use p2_gg_httbar_d261h12l121_qp, only: numerator_tmu_diagram261 => numerator_tmu
   use p2_gg_httbar_d261h12l131_qp, only: numerator_t3_diagram261 => numerator_t3
   use p2_gg_httbar_d261h12l132_qp, only: numerator_t2_diagram261 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram72, numerator_t3_diagram72, nu&
         &merator_t2_diagram72, &
          &  numerator_tmu_diagram72, &
          & 5, 4, 3, (/2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram75, numerator_t3_diagram75, nu&
         &merator_t2_diagram75, &
          &  numerator_tmu_diagram75, &
          & 5, 4, 3, (/1,2,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram261, numerator_t3_diagram261, &
         &numerator_t2_diagram261, &
          &  numerator_tmu_diagram261, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d86h12l1_qp, only: numerator_diagram86 => numerator_ninja
   use p2_gg_httbar_d86h12l121_qp, only: numerator_tmu_diagram86 => numerator_tmu
   use p2_gg_httbar_d86h12l131_qp, only: numerator_t3_diagram86 => numerator_t3
   use p2_gg_httbar_d86h12l132_qp, only: numerator_t2_diagram86 => numerator_t2
   use p2_gg_httbar_d264h12l1_qp, only: numerator_diagram264 => numerator_ninja
   use p2_gg_httbar_d264h12l121_qp, only: numerator_tmu_diagram264 => numerator_tmu
   use p2_gg_httbar_d264h12l131_qp, only: numerator_t3_diagram264 => numerator_t3
   use p2_gg_httbar_d264h12l132_qp, only: numerator_t2_diagram264 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram86, numerator_t3_diagram86, nu&
         &merator_t2_diagram86, &
          &  numerator_tmu_diagram86, &
          & 5, 4, 3, (/1,2,3,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram264, numerator_t3_diagram264, &
         &numerator_t2_diagram264, &
          &  numerator_tmu_diagram264, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
   use quadninjago_module
   use p2_gg_httbar_kinematics_qp
   use p2_gg_httbar_model_qp
   use p2_gg_httbar_d92h12l1_qp, only: numerator_diagram92 => numerator_ninja
   use p2_gg_httbar_d92h12l121_qp, only: numerator_tmu_diagram92 => numerator_tmu
   use p2_gg_httbar_d92h12l131_qp, only: numerator_t3_diagram92 => numerator_t3
   use p2_gg_httbar_d92h12l132_qp, only: numerator_t2_diagram92 => numerator_t2
   use p2_gg_httbar_d263h12l1_qp, only: numerator_diagram263 => numerator_ninja
   use p2_gg_httbar_d263h12l121_qp, only: numerator_tmu_diagram263 => numerator_tmu
   use p2_gg_httbar_d263h12l131_qp, only: numerator_t3_diagram263 => numerator_t3
   use p2_gg_httbar_d263h12l132_qp, only: numerator_t2_diagram263 => numerator_t2
   use p2_gg_httbar_globalsl1_qp, only: epspow

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
         call quadninja_diagram(numerator_diagram92, numerator_t3_diagram92, nu&
         &merator_t2_diagram92, &
          &  numerator_tmu_diagram92, &
          & 5, 4, 3, (/1,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
         call quadninja_diagram(numerator_diagram263, numerator_t3_diagram263, &
         &numerator_t2_diagram263, &
          &  numerator_tmu_diagram263, &
          & 5, 5, 4, (/1,2,3,4,5/), &
          & Vi, msq, s_mat, scale2, istopm, &
          & acc, accr, acc_ok)
            ok = ok .and. (acc_ok.eq.QUADNINJA_SUCCESS)
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
end module p2_gg_httbar_ninjah12_qp

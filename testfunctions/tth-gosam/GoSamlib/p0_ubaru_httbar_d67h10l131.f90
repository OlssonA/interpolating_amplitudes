module     p0_ubaru_httbar_d67h10l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d67h10l131.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1mu0 = 0
   integer, parameter :: ninjaidxt0mu0 = 1
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd67h10
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd67
      complex(ki), dimension (0:*), intent(inout) :: brack
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd67h10
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(23) :: acd67
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd67(1)=dotproduct(k2,ninjaE3)
      acd67(2)=dotproduct(ninjaE3,spvak2k1)
      acd67(3)=abb67(9)
      acd67(4)=dotproduct(ninjaE3,spval3k2)
      acd67(5)=abb67(11)
      acd67(6)=dotproduct(k2,ninjaA)
      acd67(7)=dotproduct(ninjaA,spvak2k1)
      acd67(8)=abb67(8)
      acd67(9)=dotproduct(l4,ninjaE3)
      acd67(10)=abb67(12)
      acd67(11)=dotproduct(ninjaA,ninjaE3)
      acd67(12)=dotproduct(ninjaA,spval3k2)
      acd67(13)=abb67(15)
      acd67(14)=abb67(10)
      acd67(15)=dotproduct(ninjaE3,spval3k1)
      acd67(16)=abb67(21)
      acd67(17)=acd67(3)*acd67(1)
      acd67(18)=acd67(5)*acd67(4)
      acd67(17)=acd67(17)+acd67(18)
      acd67(18)=acd67(2)*acd67(17)
      acd67(19)=acd67(6)*acd67(3)
      acd67(20)=acd67(12)*acd67(5)
      acd67(19)=acd67(13)+acd67(20)+acd67(19)
      acd67(19)=acd67(2)*acd67(19)
      acd67(17)=acd67(7)*acd67(17)
      acd67(20)=-2.0_ki*acd67(11)-acd67(9)
      acd67(20)=acd67(10)*acd67(20)
      acd67(21)=acd67(8)*acd67(1)
      acd67(22)=acd67(14)*acd67(4)
      acd67(23)=acd67(16)*acd67(15)
      acd67(17)=acd67(23)+acd67(22)+acd67(21)+acd67(17)+acd67(20)+acd67(19)
      brack(ninjaidxt1mu0)=acd67(18)
      brack(ninjaidxt0mu0)=acd67(17)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d67h10_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd67h10
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-2))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d67h10l131

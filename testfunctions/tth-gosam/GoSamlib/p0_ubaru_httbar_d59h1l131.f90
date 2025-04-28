module     p0_ubaru_httbar_d59h1l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d59h1l131.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd59h1
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd59
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd59h1
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(41) :: acd59
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd59(1)=dotproduct(ninjaE3,spval4k2)
      acd59(2)=abb59(8)
      acd59(3)=dotproduct(ninjaE3,spval5k2)
      acd59(4)=abb59(14)
      acd59(5)=dotproduct(ninjaA,ninjaE3)
      acd59(6)=dotproduct(ninjaE3,spvak1k2)
      acd59(7)=abb59(16)
      acd59(8)=abb59(17)
      acd59(9)=dotproduct(l4,ninjaE3)
      acd59(10)=abb59(21)
      acd59(11)=dotproduct(l5,ninjaE3)
      acd59(12)=abb59(19)
      acd59(13)=dotproduct(ninjaA,ninjaA)
      acd59(14)=dotproduct(ninjaA,spval4k2)
      acd59(15)=dotproduct(ninjaA,spval5k2)
      acd59(16)=abb59(15)
      acd59(17)=dotproduct(ninjaA,spvak1k2)
      acd59(18)=abb59(9)
      acd59(19)=dotproduct(ninjaE3,spval3k2)
      acd59(20)=abb59(10)
      acd59(21)=abb59(12)
      acd59(22)=abb59(13)
      acd59(23)=dotproduct(ninjaE3,spval5l4)
      acd59(24)=abb59(20)
      acd59(25)=dotproduct(ninjaE3,spval4l5)
      acd59(26)=abb59(24)
      acd59(27)=acd59(2)*acd59(1)
      acd59(28)=acd59(4)*acd59(3)
      acd59(27)=acd59(27)+acd59(28)
      acd59(28)=2.0_ki*acd59(5)
      acd59(29)=acd59(27)*acd59(28)
      acd59(30)=acd59(7)*acd59(6)
      acd59(31)=acd59(1)*acd59(30)
      acd59(32)=acd59(8)*acd59(6)
      acd59(33)=acd59(3)*acd59(32)
      acd59(29)=acd59(33)+acd59(29)+acd59(31)
      acd59(31)=ninjaP+acd59(13)
      acd59(31)=acd59(27)*acd59(31)
      acd59(33)=acd59(2)*acd59(28)
      acd59(30)=acd59(33)+acd59(30)
      acd59(30)=acd59(14)*acd59(30)
      acd59(33)=acd59(4)*acd59(28)
      acd59(32)=acd59(33)+acd59(32)
      acd59(32)=acd59(15)*acd59(32)
      acd59(33)=acd59(7)*acd59(1)
      acd59(34)=acd59(8)*acd59(3)
      acd59(33)=acd59(33)+acd59(34)
      acd59(33)=acd59(17)*acd59(33)
      acd59(34)=acd59(10)*acd59(9)
      acd59(35)=acd59(12)*acd59(11)
      acd59(28)=acd59(16)*acd59(28)
      acd59(36)=acd59(18)*acd59(1)
      acd59(37)=acd59(20)*acd59(19)
      acd59(38)=acd59(21)*acd59(6)
      acd59(39)=acd59(22)*acd59(3)
      acd59(40)=acd59(24)*acd59(23)
      acd59(41)=acd59(26)*acd59(25)
      acd59(28)=acd59(41)+acd59(40)+acd59(39)+acd59(38)+acd59(37)+acd59(36)+acd&
      &59(28)+acd59(35)+acd59(34)+acd59(33)+acd59(32)+acd59(30)+acd59(31)
      brack(ninjaidxt1mu0)=acd59(29)
      brack(ninjaidxt0mu0)=acd59(28)
      brack(ninjaidxt0mu2)=acd59(27)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d59h1_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd59h1
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d59h1l131

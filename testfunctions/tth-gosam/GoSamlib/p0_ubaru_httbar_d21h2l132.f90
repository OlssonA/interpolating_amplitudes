module     p0_ubaru_httbar_d21h2l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d21h2l132.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd21h2
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(30) :: acd21
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd21(1)=dotproduct(k2,ninjaE3)
      acd21(2)=abb21(12)
      acd21(3)=dotproduct(l3,ninjaE3)
      acd21(4)=abb21(15)
      acd21(5)=dotproduct(l4,ninjaE3)
      acd21(6)=abb21(19)
      acd21(7)=dotproduct(ninjaE3,spval4l3)
      acd21(8)=abb21(13)
      acd21(9)=dotproduct(ninjaE3,spvak2l3)
      acd21(10)=abb21(14)
      acd21(11)=dotproduct(ninjaE3,spval3k2)
      acd21(12)=abb21(18)
      acd21(13)=dotproduct(ninjaE3,spval4k2)
      acd21(14)=abb21(20)
      acd21(15)=dotproduct(ninjaE3,spval3k1)
      acd21(16)=abb21(21)
      acd21(17)=dotproduct(ninjaE3,spval4k1)
      acd21(18)=abb21(23)
      acd21(19)=dotproduct(ninjaE3,spval3l4)
      acd21(20)=abb21(25)
      acd21(21)=acd21(2)*acd21(1)
      acd21(22)=acd21(4)*acd21(3)
      acd21(23)=acd21(6)*acd21(5)
      acd21(24)=acd21(8)*acd21(7)
      acd21(25)=acd21(10)*acd21(9)
      acd21(26)=acd21(12)*acd21(11)
      acd21(27)=acd21(14)*acd21(13)
      acd21(28)=acd21(16)*acd21(15)
      acd21(29)=acd21(18)*acd21(17)
      acd21(30)=acd21(20)*acd21(19)
      acd21(21)=acd21(30)+acd21(29)+acd21(28)+acd21(27)+acd21(26)+acd21(25)+acd&
      &21(24)+acd21(23)+acd21(21)+acd21(22)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd21(21)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd21h2
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(42) :: acd21
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd21(1)=dotproduct(k2,ninjaA1)
      acd21(2)=abb21(12)
      acd21(3)=dotproduct(l3,ninjaA1)
      acd21(4)=abb21(15)
      acd21(5)=dotproduct(l4,ninjaA1)
      acd21(6)=abb21(19)
      acd21(7)=dotproduct(ninjaA1,spval4l3)
      acd21(8)=abb21(13)
      acd21(9)=dotproduct(ninjaA1,spvak2l3)
      acd21(10)=abb21(14)
      acd21(11)=dotproduct(ninjaA1,spval3k2)
      acd21(12)=abb21(18)
      acd21(13)=dotproduct(ninjaA1,spval4k2)
      acd21(14)=abb21(20)
      acd21(15)=dotproduct(ninjaA1,spval3k1)
      acd21(16)=abb21(21)
      acd21(17)=dotproduct(ninjaA1,spval4k1)
      acd21(18)=abb21(23)
      acd21(19)=dotproduct(ninjaA1,spval3l4)
      acd21(20)=abb21(25)
      acd21(21)=dotproduct(k2,ninjaA0)
      acd21(22)=dotproduct(l3,ninjaA0)
      acd21(23)=dotproduct(l4,ninjaA0)
      acd21(24)=dotproduct(ninjaA0,spval4l3)
      acd21(25)=dotproduct(ninjaA0,spvak2l3)
      acd21(26)=dotproduct(ninjaA0,spval3k2)
      acd21(27)=dotproduct(ninjaA0,spval4k2)
      acd21(28)=dotproduct(ninjaA0,spval3k1)
      acd21(29)=dotproduct(ninjaA0,spval4k1)
      acd21(30)=dotproduct(ninjaA0,spval3l4)
      acd21(31)=abb21(16)
      acd21(32)=acd21(1)*acd21(2)
      acd21(33)=acd21(3)*acd21(4)
      acd21(34)=acd21(5)*acd21(6)
      acd21(35)=acd21(7)*acd21(8)
      acd21(36)=acd21(9)*acd21(10)
      acd21(37)=acd21(11)*acd21(12)
      acd21(38)=acd21(13)*acd21(14)
      acd21(39)=acd21(15)*acd21(16)
      acd21(40)=acd21(17)*acd21(18)
      acd21(41)=acd21(19)*acd21(20)
      acd21(32)=acd21(41)+acd21(40)+acd21(39)+acd21(38)+acd21(37)+acd21(36)+acd&
      &21(35)+acd21(34)+acd21(32)+acd21(33)
      acd21(33)=acd21(21)*acd21(2)
      acd21(34)=acd21(22)*acd21(4)
      acd21(35)=acd21(23)*acd21(6)
      acd21(36)=acd21(24)*acd21(8)
      acd21(37)=acd21(25)*acd21(10)
      acd21(38)=acd21(26)*acd21(12)
      acd21(39)=acd21(27)*acd21(14)
      acd21(40)=acd21(28)*acd21(16)
      acd21(41)=acd21(29)*acd21(18)
      acd21(42)=acd21(30)*acd21(20)
      acd21(33)=acd21(31)+acd21(42)+acd21(41)+acd21(40)+acd21(39)+acd21(38)+acd&
      &21(37)+acd21(36)+acd21(35)+acd21(33)+acd21(34)
      brack(ninjaidxt0x0mu0)=acd21(33)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd21(32)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d21h2_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd21h2
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4+k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_ubaru_httbar_d21h2l132

module     p0_ubaru_httbar_d2h9l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d2h9l131.f90
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
      use p0_ubaru_httbar_abbrevd2h9
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd2
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd2(1)=dotproduct(k2,ninjaE3)
      acd2(2)=dotproduct(ninjaE3,spvak1k2)
      acd2(3)=abb2(12)
      acd2(4)=dotproduct(ninjaE3,spval3l5)
      acd2(5)=abb2(11)
      acd2(6)=dotproduct(ninjaE3,spval4l5)
      acd2(7)=abb2(13)
      acd2(8)=dotproduct(ninjaE3,spvak2l3)
      acd2(9)=abb2(16)
      acd2(10)=acd2(3)*acd2(1)
      acd2(11)=acd2(5)*acd2(4)
      acd2(12)=acd2(7)*acd2(6)
      acd2(13)=acd2(9)*acd2(8)
      acd2(10)=acd2(13)+acd2(12)+acd2(10)+acd2(11)
      acd2(10)=acd2(2)*acd2(10)
      brack(ninjaidxt2mu0)=acd2(10)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd2h9
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(71) :: acd2
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd2(1)=dotproduct(k2,ninjaE3)
      acd2(2)=dotproduct(ninjaE4,spvak1k2)
      acd2(3)=abb2(12)
      acd2(4)=dotproduct(k2,ninjaE4)
      acd2(5)=dotproduct(ninjaE3,spvak1k2)
      acd2(6)=dotproduct(ninjaE4,spval3l5)
      acd2(7)=abb2(11)
      acd2(8)=dotproduct(ninjaE4,spval4l5)
      acd2(9)=abb2(13)
      acd2(10)=dotproduct(ninjaE4,spvak2l3)
      acd2(11)=abb2(16)
      acd2(12)=dotproduct(ninjaE3,spval3l5)
      acd2(13)=dotproduct(ninjaE3,spval4l5)
      acd2(14)=dotproduct(ninjaE3,spvak2l3)
      acd2(15)=abb2(24)
      acd2(16)=dotproduct(k1,ninjaE3)
      acd2(17)=abb2(18)
      acd2(18)=dotproduct(k2,ninjaA)
      acd2(19)=dotproduct(ninjaA,spvak1k2)
      acd2(20)=abb2(23)
      acd2(21)=dotproduct(l5,ninjaE3)
      acd2(22)=abb2(15)
      acd2(23)=dotproduct(ninjaA,ninjaE3)
      acd2(24)=dotproduct(ninjaA,spval3l5)
      acd2(25)=dotproduct(ninjaA,spval4l5)
      acd2(26)=dotproduct(ninjaA,spvak2l3)
      acd2(27)=abb2(9)
      acd2(28)=abb2(39)
      acd2(29)=abb2(17)
      acd2(30)=dotproduct(ninjaE3,spvak1l3)
      acd2(31)=abb2(14)
      acd2(32)=dotproduct(ninjaE3,spval5k2)
      acd2(33)=abb2(31)
      acd2(34)=dotproduct(ninjaE3,spval4k2)
      acd2(35)=abb2(36)
      acd2(36)=dotproduct(ninjaE3,spval3k2)
      acd2(37)=abb2(42)
      acd2(38)=dotproduct(ninjaE3,spval5l3)
      acd2(39)=abb2(45)
      acd2(40)=dotproduct(ninjaE3,spvak1l5)
      acd2(41)=abb2(46)
      acd2(42)=dotproduct(ninjaE3,spvak2l5)
      acd2(43)=abb2(47)
      acd2(44)=dotproduct(k1,ninjaA)
      acd2(45)=dotproduct(l5,ninjaA)
      acd2(46)=dotproduct(ninjaA,ninjaA)
      acd2(47)=dotproduct(ninjaA,spvak1l3)
      acd2(48)=dotproduct(ninjaA,spval5k2)
      acd2(49)=dotproduct(ninjaA,spval4k2)
      acd2(50)=dotproduct(ninjaA,spval3k2)
      acd2(51)=dotproduct(ninjaA,spval5l3)
      acd2(52)=dotproduct(ninjaA,spvak1l5)
      acd2(53)=dotproduct(ninjaA,spvak2l5)
      acd2(54)=abb2(10)
      acd2(55)=acd2(11)*acd2(10)
      acd2(56)=acd2(9)*acd2(8)
      acd2(57)=acd2(7)*acd2(6)
      acd2(58)=acd2(3)*acd2(4)
      acd2(55)=acd2(58)+acd2(55)+acd2(56)+acd2(57)
      acd2(55)=acd2(55)*acd2(5)
      acd2(56)=acd2(11)*acd2(14)
      acd2(57)=acd2(9)*acd2(13)
      acd2(58)=acd2(7)*acd2(12)
      acd2(59)=acd2(3)*acd2(1)
      acd2(56)=acd2(56)+acd2(57)+acd2(58)+acd2(59)
      acd2(57)=acd2(56)*acd2(2)
      acd2(55)=acd2(55)+acd2(57)+acd2(15)
      acd2(56)=acd2(19)*acd2(56)
      acd2(57)=acd2(11)*acd2(26)
      acd2(58)=acd2(9)*acd2(25)
      acd2(59)=acd2(7)*acd2(24)
      acd2(60)=acd2(3)*acd2(18)
      acd2(57)=acd2(59)+acd2(57)+acd2(58)+acd2(60)+acd2(27)
      acd2(58)=acd2(5)*acd2(57)
      acd2(59)=acd2(43)*acd2(42)
      acd2(60)=acd2(41)*acd2(40)
      acd2(61)=-acd2(39)*acd2(38)
      acd2(62)=acd2(37)*acd2(36)
      acd2(63)=acd2(35)*acd2(34)
      acd2(64)=acd2(33)*acd2(32)
      acd2(65)=acd2(31)*acd2(30)
      acd2(66)=acd2(22)*acd2(21)
      acd2(67)=acd2(17)*acd2(16)
      acd2(68)=acd2(15)*acd2(23)
      acd2(69)=acd2(13)*acd2(29)
      acd2(70)=acd2(12)*acd2(28)
      acd2(71)=acd2(1)*acd2(20)
      acd2(56)=acd2(58)+acd2(56)+acd2(71)+acd2(70)+acd2(69)+2.0_ki*acd2(68)+acd&
      &2(67)+acd2(66)+acd2(65)+acd2(64)+acd2(63)+acd2(62)+acd2(61)+acd2(59)+acd&
      &2(60)
      acd2(58)=ninjaP*acd2(55)
      acd2(57)=acd2(19)*acd2(57)
      acd2(59)=acd2(43)*acd2(53)
      acd2(60)=acd2(41)*acd2(52)
      acd2(61)=-acd2(39)*acd2(51)
      acd2(62)=acd2(37)*acd2(50)
      acd2(63)=acd2(35)*acd2(49)
      acd2(64)=acd2(33)*acd2(48)
      acd2(65)=acd2(31)*acd2(47)
      acd2(66)=acd2(22)*acd2(45)
      acd2(67)=acd2(17)*acd2(44)
      acd2(68)=acd2(25)*acd2(29)
      acd2(69)=acd2(24)*acd2(28)
      acd2(70)=acd2(18)*acd2(20)
      acd2(71)=acd2(15)*acd2(46)
      acd2(57)=acd2(57)+acd2(71)+acd2(70)+acd2(69)+acd2(68)+acd2(67)+acd2(66)+a&
      &cd2(65)+acd2(64)+acd2(63)+acd2(62)+acd2(61)+acd2(60)+acd2(54)+acd2(59)+a&
      &cd2(58)
      brack(ninjaidxt1mu0)=acd2(56)
      brack(ninjaidxt0mu0)=acd2(57)
      brack(ninjaidxt0mu2)=acd2(55)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d2h9_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd2h9
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = - a(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d2h9l131

module     p2_gg_httbar_d85h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d85h0l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd85h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd85
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd85(1)=dotproduct(e2,ninjaE3)
      acd85(2)=dotproduct(ninjaE3,spvae1k2)
      acd85(3)=dotproduct(ninjaE3,spval5e1)
      acd85(4)=abb85(14)
      acd85(5)=dotproduct(ninjaE3,spval4e1)
      acd85(6)=abb85(22)
      acd85(7)=dotproduct(ninjaE3,spval3e1)
      acd85(8)=abb85(26)
      acd85(9)=dotproduct(ninjaE3,spvae1l3)
      acd85(10)=abb85(68)
      acd85(11)=acd85(4)*acd85(3)
      acd85(12)=acd85(6)*acd85(5)
      acd85(13)=acd85(8)*acd85(7)
      acd85(11)=acd85(13)+acd85(11)+acd85(12)
      acd85(11)=acd85(11)*acd85(2)
      acd85(12)=acd85(10)*acd85(9)*acd85(3)
      acd85(11)=acd85(12)+acd85(11)
      acd85(11)=acd85(1)*acd85(11)
      brack(ninjaidxt1x0mu0)=acd85(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd85h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(70) :: acd85
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd85(1)=dotproduct(e2,ninjaA1)
      acd85(2)=dotproduct(ninjaE3,spval5e1)
      acd85(3)=dotproduct(ninjaE3,spvae1k2)
      acd85(4)=abb85(14)
      acd85(5)=dotproduct(ninjaE3,spvae1l3)
      acd85(6)=abb85(68)
      acd85(7)=dotproduct(ninjaE3,spval4e1)
      acd85(8)=abb85(22)
      acd85(9)=dotproduct(ninjaE3,spval3e1)
      acd85(10)=abb85(26)
      acd85(11)=dotproduct(e2,ninjaE3)
      acd85(12)=dotproduct(ninjaA1,spval5e1)
      acd85(13)=dotproduct(ninjaA1,spvae1k2)
      acd85(14)=dotproduct(ninjaA1,spval4e1)
      acd85(15)=dotproduct(ninjaA1,spval3e1)
      acd85(16)=dotproduct(ninjaA1,spvae1l3)
      acd85(17)=dotproduct(e2,ninjaA0)
      acd85(18)=dotproduct(ninjaA0,spval5e1)
      acd85(19)=dotproduct(ninjaA0,spvae1k2)
      acd85(20)=dotproduct(ninjaA0,spval4e1)
      acd85(21)=dotproduct(ninjaA0,spval3e1)
      acd85(22)=dotproduct(ninjaA0,spvae1l3)
      acd85(23)=abb85(16)
      acd85(24)=abb85(18)
      acd85(25)=abb85(24)
      acd85(26)=abb85(47)
      acd85(27)=abb85(50)
      acd85(28)=dotproduct(ninjaA0,ninjaE3)
      acd85(29)=dotproduct(ninjaE3,spvae1e2)
      acd85(30)=abb85(49)
      acd85(31)=dotproduct(ninjaE3,spvae2e1)
      acd85(32)=abb85(32)
      acd85(33)=abb85(9)
      acd85(34)=abb85(12)
      acd85(35)=abb85(65)
      acd85(36)=abb85(23)
      acd85(37)=abb85(73)
      acd85(38)=dotproduct(ninjaE3,spvak2e1)
      acd85(39)=abb85(28)
      acd85(40)=dotproduct(ninjaE3,spvak1l3)
      acd85(41)=abb85(35)
      acd85(42)=dotproduct(ninjaE3,spvak1k2)
      acd85(43)=abb85(38)
      acd85(44)=abb85(11)
      acd85(45)=abb85(15)
      acd85(46)=abb85(25)
      acd85(47)=dotproduct(ninjaE3,spval4k1)
      acd85(48)=abb85(29)
      acd85(49)=dotproduct(ninjaE3,spval3k1)
      acd85(50)=abb85(31)
      acd85(51)=abb85(37)
      acd85(52)=dotproduct(ninjaE3,spvae1l5)
      acd85(53)=abb85(61)
      acd85(54)=acd85(8)*acd85(7)
      acd85(55)=acd85(10)*acd85(9)
      acd85(54)=acd85(54)+acd85(55)
      acd85(55)=acd85(3)*acd85(54)
      acd85(56)=acd85(2)*acd85(3)
      acd85(57)=acd85(56)*acd85(4)
      acd85(58)=acd85(6)*acd85(5)
      acd85(59)=acd85(58)*acd85(2)
      acd85(55)=acd85(59)+acd85(55)+acd85(57)
      acd85(57)=acd85(1)*acd85(55)
      acd85(54)=acd85(11)*acd85(54)
      acd85(59)=acd85(2)*acd85(11)
      acd85(60)=acd85(59)*acd85(4)
      acd85(54)=acd85(60)+acd85(54)
      acd85(60)=acd85(13)*acd85(54)
      acd85(61)=acd85(3)*acd85(11)
      acd85(62)=acd85(61)*acd85(4)
      acd85(58)=acd85(58)*acd85(11)
      acd85(58)=acd85(62)+acd85(58)
      acd85(62)=acd85(12)*acd85(58)
      acd85(63)=acd85(14)*acd85(8)*acd85(61)
      acd85(64)=acd85(61)*acd85(10)
      acd85(65)=acd85(15)*acd85(64)
      acd85(66)=acd85(59)*acd85(6)
      acd85(67)=acd85(16)*acd85(66)
      acd85(57)=acd85(67)+acd85(65)+acd85(63)+acd85(62)+acd85(57)+acd85(60)
      acd85(60)=2.0_ki*acd85(28)
      acd85(62)=acd85(30)*acd85(60)
      acd85(63)=acd85(33)*acd85(2)
      acd85(65)=acd85(36)*acd85(7)
      acd85(67)=acd85(37)*acd85(9)
      acd85(68)=acd85(39)*acd85(38)
      acd85(69)=acd85(41)*acd85(40)
      acd85(70)=acd85(43)*acd85(42)
      acd85(62)=acd85(70)+acd85(69)+acd85(68)+acd85(67)+acd85(65)+acd85(63)+acd&
      &85(62)
      acd85(62)=acd85(29)*acd85(62)
      acd85(60)=acd85(32)*acd85(60)
      acd85(63)=acd85(44)*acd85(3)
      acd85(65)=acd85(48)*acd85(47)
      acd85(67)=acd85(50)*acd85(49)
      acd85(68)=acd85(51)*acd85(5)
      acd85(69)=acd85(53)*acd85(52)
      acd85(60)=acd85(69)+acd85(68)+acd85(67)+acd85(65)+acd85(63)+acd85(60)
      acd85(60)=acd85(31)*acd85(60)
      acd85(63)=acd85(25)*acd85(7)
      acd85(65)=acd85(26)*acd85(9)
      acd85(67)=acd85(27)*acd85(5)
      acd85(63)=acd85(67)+acd85(65)+acd85(63)
      acd85(63)=acd85(11)*acd85(63)
      acd85(55)=acd85(17)*acd85(55)
      acd85(65)=acd85(45)*acd85(7)
      acd85(67)=acd85(46)*acd85(9)
      acd85(65)=acd85(67)+acd85(65)
      acd85(65)=acd85(3)*acd85(65)
      acd85(67)=acd85(20)*acd85(8)
      acd85(67)=acd85(24)+acd85(67)
      acd85(61)=acd85(61)*acd85(67)
      acd85(54)=acd85(19)*acd85(54)
      acd85(58)=acd85(18)*acd85(58)
      acd85(64)=acd85(21)*acd85(64)
      acd85(66)=acd85(22)*acd85(66)
      acd85(59)=acd85(23)*acd85(59)
      acd85(56)=acd85(34)*acd85(56)
      acd85(67)=acd85(35)*acd85(5)*acd85(2)
      acd85(54)=acd85(67)+acd85(56)+acd85(59)+acd85(66)+acd85(64)+acd85(58)+acd&
      &85(55)+acd85(54)+acd85(62)+acd85(60)+acd85(63)+acd85(61)+acd85(65)
      brack(ninjaidxt0x0mu0)=acd85(54)
      brack(ninjaidxt0x1mu0)=acd85(57)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d85h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd85h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5-k2
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d85h0l132

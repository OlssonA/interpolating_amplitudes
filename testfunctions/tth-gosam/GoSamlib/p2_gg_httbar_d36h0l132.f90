module     p2_gg_httbar_d36h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d36h0l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd36h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(60) :: acd36
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd36(1)=dotproduct(k2,ninjaE3)
      acd36(2)=abb36(17)
      acd36(3)=dotproduct(l3,ninjaE3)
      acd36(4)=abb36(15)
      acd36(5)=dotproduct(l4,ninjaE3)
      acd36(6)=abb36(70)
      acd36(7)=dotproduct(ninjaE3,spval4k2)
      acd36(8)=abb36(16)
      acd36(9)=dotproduct(ninjaE3,spval3k2)
      acd36(10)=abb36(20)
      acd36(11)=dotproduct(ninjaE3,spval4k1)
      acd36(12)=abb36(21)
      acd36(13)=dotproduct(ninjaE3,spval3k1)
      acd36(14)=abb36(22)
      acd36(15)=dotproduct(ninjaE3,spval4e2)
      acd36(16)=abb36(23)
      acd36(17)=dotproduct(ninjaE3,spvak1k2)
      acd36(18)=abb36(24)
      acd36(19)=dotproduct(ninjaE3,spval3l4)
      acd36(20)=abb36(26)
      acd36(21)=dotproduct(ninjaE3,spvak2l3)
      acd36(22)=abb36(27)
      acd36(23)=dotproduct(ninjaE3,spvae2k2)
      acd36(24)=abb36(29)
      acd36(25)=dotproduct(ninjaE3,spval4e1)
      acd36(26)=abb36(30)
      acd36(27)=dotproduct(ninjaE3,spvak1l3)
      acd36(28)=abb36(31)
      acd36(29)=dotproduct(ninjaE3,spval4l3)
      acd36(30)=abb36(34)
      acd36(31)=dotproduct(ninjaE3,spvae1k2)
      acd36(32)=abb36(35)
      acd36(33)=dotproduct(ninjaE3,spvae2l3)
      acd36(34)=abb36(38)
      acd36(35)=dotproduct(ninjaE3,spval3e2)
      acd36(36)=abb36(40)
      acd36(37)=dotproduct(ninjaE3,spvae1l3)
      acd36(38)=abb36(42)
      acd36(39)=dotproduct(ninjaE3,spval3e1)
      acd36(40)=abb36(43)
      acd36(41)=acd36(2)*acd36(1)
      acd36(42)=acd36(4)*acd36(3)
      acd36(43)=acd36(6)*acd36(5)
      acd36(44)=acd36(8)*acd36(7)
      acd36(45)=acd36(10)*acd36(9)
      acd36(46)=acd36(12)*acd36(11)
      acd36(47)=acd36(14)*acd36(13)
      acd36(48)=acd36(16)*acd36(15)
      acd36(49)=acd36(18)*acd36(17)
      acd36(50)=acd36(20)*acd36(19)
      acd36(51)=acd36(22)*acd36(21)
      acd36(52)=acd36(24)*acd36(23)
      acd36(53)=acd36(26)*acd36(25)
      acd36(54)=acd36(28)*acd36(27)
      acd36(55)=acd36(30)*acd36(29)
      acd36(56)=acd36(32)*acd36(31)
      acd36(57)=acd36(34)*acd36(33)
      acd36(58)=acd36(36)*acd36(35)
      acd36(59)=acd36(38)*acd36(37)
      acd36(60)=acd36(40)*acd36(39)
      acd36(41)=acd36(60)+acd36(59)+acd36(58)+acd36(57)+acd36(56)+acd36(55)+acd&
      &36(54)+acd36(53)+acd36(52)+acd36(51)+acd36(50)+acd36(49)+acd36(48)+acd36&
      &(47)+acd36(46)+acd36(45)+acd36(44)+acd36(43)+acd36(41)+acd36(42)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd36(41)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd36h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(82) :: acd36
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd36(1)=dotproduct(k2,ninjaA1)
      acd36(2)=abb36(17)
      acd36(3)=dotproduct(l3,ninjaA1)
      acd36(4)=abb36(15)
      acd36(5)=dotproduct(l4,ninjaA1)
      acd36(6)=abb36(70)
      acd36(7)=dotproduct(ninjaA1,spval4k2)
      acd36(8)=abb36(16)
      acd36(9)=dotproduct(ninjaA1,spval3k2)
      acd36(10)=abb36(20)
      acd36(11)=dotproduct(ninjaA1,spval4k1)
      acd36(12)=abb36(21)
      acd36(13)=dotproduct(ninjaA1,spval3k1)
      acd36(14)=abb36(22)
      acd36(15)=dotproduct(ninjaA1,spval4e2)
      acd36(16)=abb36(23)
      acd36(17)=dotproduct(ninjaA1,spvak1k2)
      acd36(18)=abb36(24)
      acd36(19)=dotproduct(ninjaA1,spval3l4)
      acd36(20)=abb36(26)
      acd36(21)=dotproduct(ninjaA1,spvak2l3)
      acd36(22)=abb36(27)
      acd36(23)=dotproduct(ninjaA1,spvae2k2)
      acd36(24)=abb36(29)
      acd36(25)=dotproduct(ninjaA1,spval4e1)
      acd36(26)=abb36(30)
      acd36(27)=dotproduct(ninjaA1,spvak1l3)
      acd36(28)=abb36(31)
      acd36(29)=dotproduct(ninjaA1,spval4l3)
      acd36(30)=abb36(34)
      acd36(31)=dotproduct(ninjaA1,spvae1k2)
      acd36(32)=abb36(35)
      acd36(33)=dotproduct(ninjaA1,spvae2l3)
      acd36(34)=abb36(38)
      acd36(35)=dotproduct(ninjaA1,spval3e2)
      acd36(36)=abb36(40)
      acd36(37)=dotproduct(ninjaA1,spvae1l3)
      acd36(38)=abb36(42)
      acd36(39)=dotproduct(ninjaA1,spval3e1)
      acd36(40)=abb36(43)
      acd36(41)=dotproduct(k2,ninjaA0)
      acd36(42)=dotproduct(l3,ninjaA0)
      acd36(43)=dotproduct(l4,ninjaA0)
      acd36(44)=dotproduct(ninjaA0,spval4k2)
      acd36(45)=dotproduct(ninjaA0,spval3k2)
      acd36(46)=dotproduct(ninjaA0,spval4k1)
      acd36(47)=dotproduct(ninjaA0,spval3k1)
      acd36(48)=dotproduct(ninjaA0,spval4e2)
      acd36(49)=dotproduct(ninjaA0,spvak1k2)
      acd36(50)=dotproduct(ninjaA0,spval3l4)
      acd36(51)=dotproduct(ninjaA0,spvak2l3)
      acd36(52)=dotproduct(ninjaA0,spvae2k2)
      acd36(53)=dotproduct(ninjaA0,spval4e1)
      acd36(54)=dotproduct(ninjaA0,spvak1l3)
      acd36(55)=dotproduct(ninjaA0,spval4l3)
      acd36(56)=dotproduct(ninjaA0,spvae1k2)
      acd36(57)=dotproduct(ninjaA0,spvae2l3)
      acd36(58)=dotproduct(ninjaA0,spval3e2)
      acd36(59)=dotproduct(ninjaA0,spvae1l3)
      acd36(60)=dotproduct(ninjaA0,spval3e1)
      acd36(61)=abb36(19)
      acd36(62)=acd36(1)*acd36(2)
      acd36(63)=acd36(3)*acd36(4)
      acd36(64)=acd36(5)*acd36(6)
      acd36(65)=acd36(7)*acd36(8)
      acd36(66)=acd36(9)*acd36(10)
      acd36(67)=acd36(11)*acd36(12)
      acd36(68)=acd36(13)*acd36(14)
      acd36(69)=acd36(15)*acd36(16)
      acd36(70)=acd36(17)*acd36(18)
      acd36(71)=acd36(19)*acd36(20)
      acd36(72)=acd36(21)*acd36(22)
      acd36(73)=acd36(23)*acd36(24)
      acd36(74)=acd36(25)*acd36(26)
      acd36(75)=acd36(27)*acd36(28)
      acd36(76)=acd36(29)*acd36(30)
      acd36(77)=acd36(31)*acd36(32)
      acd36(78)=acd36(33)*acd36(34)
      acd36(79)=acd36(35)*acd36(36)
      acd36(80)=acd36(37)*acd36(38)
      acd36(81)=acd36(39)*acd36(40)
      acd36(62)=acd36(81)+acd36(80)+acd36(79)+acd36(78)+acd36(77)+acd36(76)+acd&
      &36(75)+acd36(74)+acd36(73)+acd36(72)+acd36(71)+acd36(70)+acd36(69)+acd36&
      &(68)+acd36(67)+acd36(66)+acd36(65)+acd36(64)+acd36(62)+acd36(63)
      acd36(63)=acd36(41)*acd36(2)
      acd36(64)=acd36(42)*acd36(4)
      acd36(65)=acd36(43)*acd36(6)
      acd36(66)=acd36(44)*acd36(8)
      acd36(67)=acd36(45)*acd36(10)
      acd36(68)=acd36(46)*acd36(12)
      acd36(69)=acd36(47)*acd36(14)
      acd36(70)=acd36(48)*acd36(16)
      acd36(71)=acd36(49)*acd36(18)
      acd36(72)=acd36(50)*acd36(20)
      acd36(73)=acd36(51)*acd36(22)
      acd36(74)=acd36(52)*acd36(24)
      acd36(75)=acd36(53)*acd36(26)
      acd36(76)=acd36(54)*acd36(28)
      acd36(77)=acd36(55)*acd36(30)
      acd36(78)=acd36(56)*acd36(32)
      acd36(79)=acd36(57)*acd36(34)
      acd36(80)=acd36(58)*acd36(36)
      acd36(81)=acd36(59)*acd36(38)
      acd36(82)=acd36(60)*acd36(40)
      acd36(63)=acd36(61)+acd36(82)+acd36(81)+acd36(80)+acd36(79)+acd36(78)+acd&
      &36(77)+acd36(76)+acd36(75)+acd36(74)+acd36(73)+acd36(72)+acd36(71)+acd36&
      &(70)+acd36(69)+acd36(68)+acd36(67)+acd36(66)+acd36(65)+acd36(63)+acd36(6&
      &4)
      brack(ninjaidxt0x0mu0)=acd36(63)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd36(62)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d36h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd36h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = - a0(0:3)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d36h0l132

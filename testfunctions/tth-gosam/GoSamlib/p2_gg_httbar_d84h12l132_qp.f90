module     p2_gg_httbar_d84h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d84h12l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd84h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd84
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd84(1)=dotproduct(e1,ninjaE3)
      acd84(2)=dotproduct(e2,ninjaE3)
      acd84(3)=dotproduct(ninjaE3,spvak2l5)
      acd84(4)=abb84(23)
      acd84(5)=dotproduct(ninjaE3,spval3l5)
      acd84(6)=abb84(27)
      acd84(7)=dotproduct(ninjaE3,spvak2l3)
      acd84(8)=abb84(32)
      acd84(9)=dotproduct(ninjaE3,spvak2l4)
      acd84(10)=abb84(43)
      acd84(11)=acd84(4)*acd84(3)
      acd84(12)=acd84(6)*acd84(5)
      acd84(13)=acd84(8)*acd84(7)
      acd84(14)=acd84(10)*acd84(9)
      acd84(11)=acd84(14)+acd84(13)+acd84(11)+acd84(12)
      acd84(11)=acd84(11)*acd84(2)*acd84(1)
      brack(ninjaidxt1x0mu0)=acd84(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd84h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(113) :: acd84
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd84(1)=dotproduct(e1,ninjaA1)
      acd84(2)=dotproduct(e2,ninjaE3)
      acd84(3)=dotproduct(ninjaE3,spvak2l5)
      acd84(4)=abb84(23)
      acd84(5)=dotproduct(ninjaE3,spvak2l4)
      acd84(6)=abb84(43)
      acd84(7)=dotproduct(ninjaE3,spval3l5)
      acd84(8)=abb84(27)
      acd84(9)=dotproduct(ninjaE3,spvak2l3)
      acd84(10)=abb84(32)
      acd84(11)=dotproduct(e1,ninjaE3)
      acd84(12)=dotproduct(e2,ninjaA1)
      acd84(13)=dotproduct(ninjaA1,spvak2l5)
      acd84(14)=dotproduct(ninjaA1,spvak2l4)
      acd84(15)=dotproduct(ninjaA1,spval3l5)
      acd84(16)=dotproduct(ninjaA1,spvak2l3)
      acd84(17)=dotproduct(k1,ninjaE3)
      acd84(18)=dotproduct(ninjaE3,spvae2e1)
      acd84(19)=abb84(29)
      acd84(20)=dotproduct(ninjaE3,spvae1e2)
      acd84(21)=abb84(44)
      acd84(22)=dotproduct(k2,ninjaE3)
      acd84(23)=abb84(12)
      acd84(24)=abb84(15)
      acd84(25)=abb84(10)
      acd84(26)=abb84(21)
      acd84(27)=abb84(48)
      acd84(28)=dotproduct(l5,ninjaE3)
      acd84(29)=abb84(14)
      acd84(30)=abb84(70)
      acd84(31)=abb84(39)
      acd84(32)=abb84(68)
      acd84(33)=dotproduct(e1,ninjaA0)
      acd84(34)=dotproduct(e2,ninjaA0)
      acd84(35)=dotproduct(ninjaA0,spvak2l5)
      acd84(36)=dotproduct(ninjaA0,spvak2l4)
      acd84(37)=dotproduct(ninjaA0,spval3l5)
      acd84(38)=dotproduct(ninjaA0,spvak2l3)
      acd84(39)=abb84(9)
      acd84(40)=dotproduct(ninjaA0,ninjaE3)
      acd84(41)=abb84(60)
      acd84(42)=abb84(24)
      acd84(43)=dotproduct(ninjaE3,spval5l3)
      acd84(44)=abb84(33)
      acd84(45)=abb84(64)
      acd84(46)=dotproduct(ninjaE3,spvae2k1)
      acd84(47)=abb84(28)
      acd84(48)=dotproduct(ninjaE3,spvak1e2)
      acd84(49)=abb84(30)
      acd84(50)=dotproduct(ninjaE3,spval3k2)
      acd84(51)=abb84(40)
      acd84(52)=dotproduct(ninjaE3,spval5l4)
      acd84(53)=abb84(56)
      acd84(54)=dotproduct(ninjaE3,spvae2l5)
      acd84(55)=abb84(51)
      acd84(56)=dotproduct(ninjaE3,spvae2l4)
      acd84(57)=abb84(54)
      acd84(58)=dotproduct(ninjaE3,spvae2l3)
      acd84(59)=abb84(61)
      acd84(60)=dotproduct(ninjaE3,spvak2e2)
      acd84(61)=abb84(72)
      acd84(62)=dotproduct(ninjaE3,spvae2k2)
      acd84(63)=abb84(81)
      acd84(64)=dotproduct(ninjaE3,spval3e2)
      acd84(65)=abb84(83)
      acd84(66)=abb84(59)
      acd84(67)=abb84(41)
      acd84(68)=dotproduct(ninjaE3,spvak2e1)
      acd84(69)=abb84(16)
      acd84(70)=abb84(25)
      acd84(71)=abb84(62)
      acd84(72)=dotproduct(ninjaE3,spvae1k1)
      acd84(73)=abb84(34)
      acd84(74)=abb84(79)
      acd84(75)=dotproduct(ninjaE3,spvak1e1)
      acd84(76)=abb84(45)
      acd84(77)=abb84(50)
      acd84(78)=dotproduct(ninjaE3,spvae1l5)
      acd84(79)=abb84(65)
      acd84(80)=dotproduct(ninjaE3,spvae1l4)
      acd84(81)=abb84(78)
      acd84(82)=dotproduct(ninjaE3,spvae1l3)
      acd84(83)=abb84(84)
      acd84(84)=dotproduct(ninjaE3,spval3e1)
      acd84(85)=abb84(85)
      acd84(86)=abb84(20)
      acd84(87)=abb84(52)
      acd84(88)=abb84(69)
      acd84(89)=abb84(67)
      acd84(90)=abb84(57)
      acd84(91)=abb84(76)
      acd84(92)=abb84(38)
      acd84(93)=acd84(13)*acd84(4)
      acd84(94)=acd84(14)*acd84(6)
      acd84(95)=acd84(15)*acd84(8)
      acd84(96)=acd84(16)*acd84(10)
      acd84(93)=acd84(96)+acd84(95)+acd84(94)+acd84(93)
      acd84(94)=acd84(2)*acd84(11)
      acd84(93)=acd84(94)*acd84(93)
      acd84(95)=acd84(4)*acd84(3)
      acd84(96)=acd84(6)*acd84(5)
      acd84(97)=acd84(8)*acd84(7)
      acd84(98)=acd84(10)*acd84(9)
      acd84(95)=acd84(98)+acd84(95)+acd84(96)+acd84(97)
      acd84(96)=acd84(95)*acd84(2)
      acd84(97)=acd84(1)*acd84(96)
      acd84(95)=acd84(95)*acd84(11)
      acd84(98)=acd84(12)*acd84(95)
      acd84(93)=acd84(97)+acd84(98)+acd84(93)
      acd84(97)=acd84(23)*acd84(22)
      acd84(98)=acd84(29)*acd84(28)
      acd84(99)=2.0_ki*acd84(40)
      acd84(100)=acd84(41)*acd84(99)
      acd84(101)=acd84(42)*acd84(3)
      acd84(102)=acd84(44)*acd84(43)
      acd84(103)=acd84(45)*acd84(7)
      acd84(104)=acd84(47)*acd84(46)
      acd84(105)=acd84(49)*acd84(48)
      acd84(106)=-acd84(51)*acd84(50)
      acd84(107)=acd84(53)*acd84(52)
      acd84(108)=-acd84(55)*acd84(54)
      acd84(109)=acd84(57)*acd84(56)
      acd84(110)=acd84(59)*acd84(58)
      acd84(111)=acd84(61)*acd84(60)
      acd84(112)=-acd84(63)*acd84(62)
      acd84(113)=acd84(65)*acd84(64)
      acd84(97)=acd84(113)+acd84(112)+acd84(111)+acd84(110)+acd84(109)+acd84(10&
      &8)+acd84(107)+acd84(106)+acd84(105)+acd84(104)+acd84(103)+acd84(102)+acd&
      &84(101)+acd84(100)+acd84(98)+acd84(97)
      acd84(97)=acd84(11)*acd84(97)
      acd84(98)=acd84(24)*acd84(22)
      acd84(100)=acd84(30)*acd84(28)
      acd84(101)=acd84(66)*acd84(99)
      acd84(102)=acd84(67)*acd84(3)
      acd84(103)=acd84(69)*acd84(68)
      acd84(104)=acd84(70)*acd84(43)
      acd84(105)=acd84(71)*acd84(7)
      acd84(106)=acd84(73)*acd84(72)
      acd84(107)=acd84(74)*acd84(50)
      acd84(108)=acd84(76)*acd84(75)
      acd84(109)=acd84(77)*acd84(52)
      acd84(110)=acd84(79)*acd84(78)
      acd84(111)=acd84(81)*acd84(80)
      acd84(112)=acd84(83)*acd84(82)
      acd84(113)=-acd84(85)*acd84(84)
      acd84(98)=acd84(113)+acd84(112)+acd84(111)+acd84(110)+acd84(109)+acd84(10&
      &8)+acd84(107)+acd84(106)+acd84(105)+acd84(104)+acd84(103)+acd84(102)+acd&
      &84(101)+acd84(100)+acd84(98)
      acd84(98)=acd84(2)*acd84(98)
      acd84(100)=acd84(86)*acd84(3)
      acd84(101)=acd84(88)*acd84(7)
      acd84(102)=acd84(89)*acd84(18)
      acd84(103)=acd84(91)*acd84(20)
      acd84(100)=acd84(103)+acd84(102)+acd84(101)+acd84(100)
      acd84(100)=acd84(99)*acd84(100)
      acd84(101)=acd84(35)*acd84(4)
      acd84(102)=acd84(36)*acd84(6)
      acd84(103)=acd84(37)*acd84(8)
      acd84(104)=acd84(38)*acd84(10)
      acd84(101)=acd84(39)+acd84(104)+acd84(103)+acd84(102)+acd84(101)
      acd84(94)=acd84(94)*acd84(101)
      acd84(96)=acd84(33)*acd84(96)
      acd84(95)=acd84(34)*acd84(95)
      acd84(101)=acd84(25)*acd84(3)
      acd84(102)=acd84(26)*acd84(5)
      acd84(103)=acd84(27)*acd84(9)
      acd84(101)=acd84(103)+acd84(102)+acd84(101)
      acd84(101)=acd84(22)*acd84(101)
      acd84(102)=-acd84(19)*acd84(18)
      acd84(103)=-acd84(21)*acd84(20)
      acd84(102)=acd84(103)+acd84(102)
      acd84(103)=acd84(17)-acd84(22)
      acd84(102)=acd84(103)*acd84(102)
      acd84(103)=acd84(31)*acd84(3)
      acd84(104)=acd84(32)*acd84(7)
      acd84(103)=acd84(104)+acd84(103)
      acd84(103)=acd84(28)*acd84(103)
      acd84(104)=-acd84(5)*acd84(99)
      acd84(105)=-acd84(52)*acd84(3)
      acd84(104)=acd84(104)+acd84(105)
      acd84(104)=acd84(87)*acd84(104)
      acd84(99)=-acd84(9)*acd84(99)
      acd84(105)=-acd84(43)*acd84(3)
      acd84(99)=acd84(99)+acd84(105)
      acd84(99)=acd84(90)*acd84(99)
      acd84(105)=acd84(92)*acd84(50)*acd84(3)
      acd84(94)=acd84(105)+acd84(99)+acd84(104)+acd84(96)+acd84(95)+acd84(97)+a&
      &cd84(98)+acd84(94)+acd84(100)+acd84(101)+acd84(103)+acd84(102)
      brack(ninjaidxt0x0mu0)=acd84(94)
      brack(ninjaidxt0x1mu0)=acd84(93)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d84h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd84h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2
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
end module     p2_gg_httbar_d84h12l132_qp

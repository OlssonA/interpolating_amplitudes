module     p2_gg_httbar_d50h8l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d50h8l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(70) :: acd50
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd50(1)=dotproduct(k2,ninjaA)
      acd50(2)=dotproduct(k2,ninjaE3)
      acd50(3)=abb50(10)
      acd50(4)=dotproduct(ninjaE3,spval3k2)
      acd50(5)=abb50(18)
      acd50(6)=dotproduct(ninjaE3,spvak2l3)
      acd50(7)=abb50(24)
      acd50(8)=dotproduct(ninjaA,spval3k2)
      acd50(9)=dotproduct(ninjaA,spvak2l3)
      acd50(10)=abb50(13)
      acd50(11)=dotproduct(ninjaA,spvak1k2)
      acd50(12)=dotproduct(ninjaE3,spvak2k1)
      acd50(13)=abb50(9)
      acd50(14)=dotproduct(ninjaE3,spval3k1)
      acd50(15)=abb50(39)
      acd50(16)=dotproduct(ninjaA,spvak2k1)
      acd50(17)=dotproduct(ninjaE3,spvak1k2)
      acd50(18)=dotproduct(ninjaE3,spvak1l3)
      acd50(19)=abb50(32)
      acd50(20)=dotproduct(ninjaA,spval4k2)
      acd50(21)=dotproduct(ninjaE3,spvak2l5)
      acd50(22)=abb50(17)
      acd50(23)=abb50(29)
      acd50(24)=dotproduct(ninjaA,spval4k1)
      acd50(25)=dotproduct(ninjaE3,spvak1l5)
      acd50(26)=dotproduct(ninjaA,spval3k1)
      acd50(27)=abb50(37)
      acd50(28)=dotproduct(ninjaA,spvak1l5)
      acd50(29)=dotproduct(ninjaE3,spval4k1)
      acd50(30)=dotproduct(ninjaA,spvak2l5)
      acd50(31)=dotproduct(ninjaE3,spval4k2)
      acd50(32)=dotproduct(ninjaA,spvak1l3)
      acd50(33)=abb50(26)
      acd50(34)=abb50(25)
      acd50(35)=abb50(11)
      acd50(36)=abb50(12)
      acd50(37)=abb50(14)
      acd50(38)=abb50(15)
      acd50(39)=abb50(21)
      acd50(40)=abb50(20)
      acd50(41)=abb50(19)
      acd50(42)=abb50(23)
      acd50(43)=acd50(5)*acd50(4)
      acd50(44)=acd50(7)*acd50(6)
      acd50(43)=acd50(43)+acd50(44)
      acd50(44)=acd50(3)*acd50(2)
      acd50(44)=2.0_ki*acd50(44)+acd50(43)
      acd50(44)=acd50(1)*acd50(44)
      acd50(45)=acd50(21)*acd50(27)
      acd50(46)=acd50(5)*acd50(2)
      acd50(46)=-acd50(45)+acd50(46)
      acd50(46)=acd50(8)*acd50(46)
      acd50(47)=acd50(7)*acd50(2)
      acd50(48)=-acd50(31)*acd50(23)
      acd50(47)=acd50(48)+acd50(47)
      acd50(47)=acd50(9)*acd50(47)
      acd50(48)=acd50(13)*acd50(12)
      acd50(49)=acd50(15)*acd50(14)
      acd50(48)=-acd50(48)-acd50(49)
      acd50(48)=acd50(11)*acd50(48)
      acd50(49)=acd50(13)*acd50(17)
      acd50(50)=acd50(19)*acd50(18)
      acd50(49)=acd50(49)+acd50(50)
      acd50(50)=-acd50(16)*acd50(49)
      acd50(51)=acd50(6)*acd50(23)
      acd50(52)=acd50(21)*acd50(22)
      acd50(51)=acd50(51)+acd50(52)
      acd50(52)=-acd50(20)*acd50(51)
      acd50(53)=acd50(18)*acd50(23)
      acd50(54)=acd50(25)*acd50(22)
      acd50(53)=acd50(53)+acd50(54)
      acd50(54)=acd50(24)*acd50(53)
      acd50(55)=acd50(25)*acd50(27)
      acd50(56)=acd50(15)*acd50(17)
      acd50(55)=acd50(55)-acd50(56)
      acd50(56)=acd50(26)*acd50(55)
      acd50(57)=acd50(14)*acd50(27)
      acd50(58)=acd50(29)*acd50(22)
      acd50(57)=acd50(57)+acd50(58)
      acd50(57)=acd50(28)*acd50(57)
      acd50(58)=acd50(4)*acd50(27)
      acd50(59)=-acd50(31)*acd50(22)
      acd50(58)=-acd50(58)+acd50(59)
      acd50(58)=acd50(30)*acd50(58)
      acd50(59)=acd50(19)*acd50(12)
      acd50(60)=acd50(29)*acd50(23)
      acd50(59)=acd50(60)-acd50(59)
      acd50(59)=acd50(32)*acd50(59)
      acd50(60)=acd50(10)*acd50(2)
      acd50(61)=acd50(33)*acd50(17)
      acd50(62)=acd50(34)*acd50(12)
      acd50(63)=acd50(35)*acd50(31)
      acd50(64)=acd50(36)*acd50(29)
      acd50(65)=acd50(37)*acd50(14)
      acd50(66)=acd50(38)*acd50(4)
      acd50(67)=acd50(39)*acd50(25)
      acd50(68)=acd50(40)*acd50(21)
      acd50(69)=acd50(41)*acd50(6)
      acd50(70)=acd50(42)*acd50(18)
      acd50(44)=acd50(70)+acd50(69)+acd50(68)+acd50(67)+acd50(66)+acd50(65)+acd&
      &50(64)+acd50(63)+acd50(62)+acd50(61)+acd50(60)+acd50(59)+acd50(58)+acd50&
      &(57)+acd50(56)+acd50(54)+acd50(52)+acd50(50)+acd50(48)+acd50(47)+acd50(4&
      &6)+acd50(44)
      acd50(43)=acd50(2)*acd50(43)
      acd50(46)=-acd50(12)*acd50(49)
      acd50(47)=acd50(14)*acd50(55)
      acd50(48)=acd50(29)*acd50(53)
      acd50(49)=-acd50(31)*acd50(51)
      acd50(45)=-acd50(4)*acd50(45)
      acd50(50)=acd50(3)*acd50(2)**2
      acd50(43)=acd50(50)+acd50(49)+acd50(48)+acd50(45)+acd50(47)+acd50(46)+acd&
      &50(43)
      brack(ninjaidxt3mu0)=acd50(43)
      brack(ninjaidxt2mu0)=acd50(44)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d50h8_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d50h8l131_qp

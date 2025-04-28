module     p2_gg_httbar_d10h12l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d10h12l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd10h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(21) :: acd10
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd10(1)=dotproduct(k2,ninjaE3)
      acd10(2)=dotproduct(ninjaE3,spvak2l3)
      acd10(3)=abb10(11)
      acd10(4)=dotproduct(ninjaE3,spvak2l5)
      acd10(5)=abb10(17)
      acd10(6)=dotproduct(ninjaE3,spvak2l4)
      acd10(7)=abb10(22)
      acd10(8)=dotproduct(ninjaE3,spvak1l3)
      acd10(9)=dotproduct(ninjaE3,spvak2k1)
      acd10(10)=abb10(12)
      acd10(11)=dotproduct(ninjaE3,spvak1l5)
      acd10(12)=abb10(27)
      acd10(13)=dotproduct(ninjaE3,spvak1l4)
      acd10(14)=abb10(28)
      acd10(15)=dotproduct(ninjaE3,spval3k2)
      acd10(16)=abb10(21)
      acd10(17)=dotproduct(ninjaE3,spval3k1)
      acd10(18)=acd10(3)*acd10(2)
      acd10(19)=acd10(5)*acd10(4)
      acd10(20)=acd10(7)*acd10(6)
      acd10(18)=acd10(20)+acd10(18)+acd10(19)
      acd10(18)=acd10(1)*acd10(18)
      acd10(19)=acd10(10)*acd10(8)
      acd10(20)=acd10(12)*acd10(11)
      acd10(21)=acd10(14)*acd10(13)
      acd10(19)=acd10(21)+acd10(20)+acd10(19)
      acd10(19)=acd10(9)*acd10(19)
      acd10(20)=acd10(15)*acd10(4)
      acd10(21)=-acd10(17)*acd10(11)
      acd10(20)=acd10(21)+acd10(20)
      acd10(20)=acd10(16)*acd10(20)
      acd10(18)=acd10(19)+acd10(18)+acd10(20)
      brack(ninjaidxt2mu0)=acd10(18)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd10h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(64) :: acd10
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd10(1)=dotproduct(k2,ninjaE3)
      acd10(2)=dotproduct(ninjaE4,spvak2l3)
      acd10(3)=abb10(11)
      acd10(4)=dotproduct(ninjaE4,spvak2l5)
      acd10(5)=abb10(17)
      acd10(6)=dotproduct(ninjaE4,spvak2l4)
      acd10(7)=abb10(22)
      acd10(8)=dotproduct(k2,ninjaE4)
      acd10(9)=dotproduct(ninjaE3,spvak2l3)
      acd10(10)=dotproduct(ninjaE3,spvak2l5)
      acd10(11)=dotproduct(ninjaE3,spvak2l4)
      acd10(12)=dotproduct(ninjaE3,spvak1l4)
      acd10(13)=dotproduct(ninjaE4,spvak2k1)
      acd10(14)=abb10(28)
      acd10(15)=dotproduct(ninjaE3,spvak1l3)
      acd10(16)=abb10(12)
      acd10(17)=dotproduct(ninjaE3,spvak2k1)
      acd10(18)=dotproduct(ninjaE4,spvak1l4)
      acd10(19)=dotproduct(ninjaE4,spvak1l3)
      acd10(20)=dotproduct(ninjaE4,spvak1l5)
      acd10(21)=abb10(27)
      acd10(22)=dotproduct(ninjaE4,spval3k2)
      acd10(23)=abb10(21)
      acd10(24)=dotproduct(ninjaE3,spval3k2)
      acd10(25)=dotproduct(ninjaE3,spvak1l5)
      acd10(26)=dotproduct(ninjaE4,spval3k1)
      acd10(27)=dotproduct(ninjaE3,spval3k1)
      acd10(28)=dotproduct(k2,ninjaA)
      acd10(29)=dotproduct(ninjaA,spvak2l3)
      acd10(30)=dotproduct(ninjaA,spvak2l5)
      acd10(31)=dotproduct(ninjaA,spvak2l4)
      acd10(32)=abb10(19)
      acd10(33)=dotproduct(ninjaA,spvak1l4)
      acd10(34)=dotproduct(ninjaA,spvak1l3)
      acd10(35)=dotproduct(ninjaA,spvak2k1)
      acd10(36)=dotproduct(ninjaA,spval3k2)
      acd10(37)=dotproduct(ninjaA,spvak1l5)
      acd10(38)=dotproduct(ninjaA,spval3k1)
      acd10(39)=abb10(9)
      acd10(40)=abb10(10)
      acd10(41)=abb10(14)
      acd10(42)=abb10(18)
      acd10(43)=abb10(13)
      acd10(44)=abb10(16)
      acd10(45)=abb10(20)
      acd10(46)=abb10(25)
      acd10(47)=abb10(26)
      acd10(48)=abb10(15)
      acd10(49)=acd10(20)*acd10(27)
      acd10(50)=acd10(4)*acd10(24)
      acd10(51)=acd10(25)*acd10(26)
      acd10(52)=acd10(10)*acd10(22)
      acd10(49)=-acd10(50)+acd10(49)+acd10(51)-acd10(52)
      acd10(49)=acd10(49)*acd10(23)
      acd10(50)=acd10(7)*acd10(6)
      acd10(51)=acd10(5)*acd10(4)
      acd10(52)=acd10(3)*acd10(2)
      acd10(50)=acd10(52)+acd10(50)+acd10(51)
      acd10(50)=acd10(50)*acd10(1)
      acd10(51)=acd10(21)*acd10(20)
      acd10(52)=acd10(16)*acd10(19)
      acd10(53)=acd10(14)*acd10(18)
      acd10(51)=acd10(53)+acd10(51)+acd10(52)
      acd10(51)=acd10(51)*acd10(17)
      acd10(52)=acd10(7)*acd10(11)
      acd10(53)=acd10(5)*acd10(10)
      acd10(54)=acd10(3)*acd10(9)
      acd10(52)=acd10(54)+acd10(52)+acd10(53)
      acd10(53)=acd10(52)*acd10(8)
      acd10(54)=acd10(21)*acd10(25)
      acd10(55)=acd10(16)*acd10(15)
      acd10(56)=acd10(14)*acd10(12)
      acd10(54)=acd10(56)+acd10(54)+acd10(55)
      acd10(55)=acd10(54)*acd10(13)
      acd10(49)=acd10(53)+acd10(55)-acd10(49)+acd10(50)+acd10(51)
      acd10(50)=-acd10(37)*acd10(27)
      acd10(51)=acd10(30)*acd10(24)
      acd10(53)=-acd10(25)*acd10(38)
      acd10(55)=acd10(10)*acd10(36)
      acd10(50)=acd10(55)+acd10(53)+acd10(50)+acd10(51)
      acd10(50)=acd10(23)*acd10(50)
      acd10(51)=acd10(35)*acd10(54)
      acd10(52)=acd10(28)*acd10(52)
      acd10(53)=acd10(21)*acd10(37)
      acd10(54)=acd10(16)*acd10(34)
      acd10(55)=acd10(14)*acd10(33)
      acd10(53)=acd10(53)+acd10(54)+acd10(55)+acd10(42)
      acd10(54)=acd10(17)*acd10(53)
      acd10(55)=acd10(7)*acd10(31)
      acd10(56)=acd10(5)*acd10(30)
      acd10(55)=acd10(32)+acd10(55)+acd10(56)
      acd10(56)=acd10(3)*acd10(29)
      acd10(56)=acd10(56)+acd10(55)
      acd10(56)=acd10(1)*acd10(56)
      acd10(57)=acd10(27)*acd10(47)
      acd10(58)=acd10(24)*acd10(45)
      acd10(59)=acd10(15)*acd10(41)
      acd10(60)=acd10(12)*acd10(40)
      acd10(61)=acd10(11)*acd10(44)
      acd10(62)=acd10(9)*acd10(39)
      acd10(63)=acd10(25)*acd10(46)
      acd10(64)=acd10(10)*acd10(43)
      acd10(50)=acd10(50)+acd10(56)+acd10(54)+acd10(64)+acd10(63)+acd10(62)+acd&
      &10(61)+acd10(60)+acd10(59)+acd10(57)+acd10(58)+acd10(52)+acd10(51)
      acd10(51)=ninjaP*acd10(49)
      acd10(52)=acd10(35)*acd10(53)
      acd10(53)=acd10(28)*acd10(55)
      acd10(54)=-acd10(37)*acd10(38)
      acd10(55)=acd10(30)*acd10(36)
      acd10(54)=acd10(54)+acd10(55)
      acd10(54)=acd10(23)*acd10(54)
      acd10(55)=acd10(3)*acd10(28)
      acd10(55)=acd10(55)+acd10(39)
      acd10(55)=acd10(29)*acd10(55)
      acd10(56)=acd10(38)*acd10(47)
      acd10(57)=acd10(36)*acd10(45)
      acd10(58)=acd10(34)*acd10(41)
      acd10(59)=acd10(33)*acd10(40)
      acd10(60)=acd10(31)*acd10(44)
      acd10(61)=acd10(37)*acd10(46)
      acd10(62)=acd10(30)*acd10(43)
      acd10(51)=acd10(51)+acd10(54)+acd10(62)+acd10(61)+acd10(60)+acd10(59)+acd&
      &10(58)+acd10(57)+acd10(48)+acd10(56)+acd10(52)+acd10(53)+acd10(55)
      brack(ninjaidxt1mu0)=acd10(50)
      brack(ninjaidxt0mu0)=acd10(51)
      brack(ninjaidxt0mu2)=acd10(49)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d10h12_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd10h12_qp
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
end module     p2_gg_httbar_d10h12l131_qp

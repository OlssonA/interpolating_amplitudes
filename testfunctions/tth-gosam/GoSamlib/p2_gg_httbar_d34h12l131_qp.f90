module     p2_gg_httbar_d34h12l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d34h12l131_qp.f90
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
      use p2_gg_httbar_abbrevd34h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd34
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd34(1)=dotproduct(ninjaE3,spvak2e1)
      acd34(2)=dotproduct(ninjaE3,spvae1k2)
      acd34(3)=abb34(13)
      acd34(4)=dotproduct(ninjaE3,spvae1l5)
      acd34(5)=abb34(26)
      acd34(6)=dotproduct(ninjaE3,spvae1e2)
      acd34(7)=abb34(23)
      acd34(8)=dotproduct(ninjaE3,spval3e1)
      acd34(9)=abb34(20)
      acd34(10)=abb34(19)
      acd34(11)=abb34(24)
      acd34(12)=dotproduct(ninjaE3,spvae1l4)
      acd34(13)=dotproduct(ninjaE3,spvae2e1)
      acd34(14)=abb34(38)
      acd34(15)=dotproduct(ninjaE3,spval5e1)
      acd34(16)=abb34(52)
      acd34(17)=dotproduct(ninjaE3,spvae1l3)
      acd34(18)=abb34(50)
      acd34(19)=abb34(48)
      acd34(20)=acd34(3)*acd34(2)
      acd34(21)=acd34(5)*acd34(4)
      acd34(22)=acd34(7)*acd34(6)
      acd34(20)=acd34(22)+acd34(20)+acd34(21)
      acd34(20)=acd34(1)*acd34(20)
      acd34(21)=acd34(9)*acd34(2)
      acd34(22)=acd34(10)*acd34(4)
      acd34(23)=acd34(11)*acd34(6)
      acd34(21)=acd34(23)+acd34(22)+acd34(21)
      acd34(21)=acd34(8)*acd34(21)
      acd34(22)=-acd34(14)*acd34(13)
      acd34(23)=-acd34(16)*acd34(15)
      acd34(22)=acd34(23)+acd34(22)
      acd34(22)=acd34(12)*acd34(22)
      acd34(23)=-acd34(18)*acd34(13)
      acd34(24)=-acd34(19)*acd34(15)
      acd34(23)=acd34(24)+acd34(23)
      acd34(23)=acd34(17)*acd34(23)
      acd34(20)=acd34(21)+acd34(20)+acd34(23)+acd34(22)
      brack(ninjaidxt2mu0)=acd34(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd34h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(67) :: acd34
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd34(1)=dotproduct(ninjaE3,spvae1k2)
      acd34(2)=dotproduct(ninjaE4,spvak2e1)
      acd34(3)=abb34(13)
      acd34(4)=dotproduct(ninjaE4,spval3e1)
      acd34(5)=abb34(20)
      acd34(6)=dotproduct(ninjaE3,spvae1l3)
      acd34(7)=dotproduct(ninjaE4,spvae2e1)
      acd34(8)=abb34(50)
      acd34(9)=dotproduct(ninjaE4,spval5e1)
      acd34(10)=abb34(48)
      acd34(11)=dotproduct(ninjaE3,spvak2e1)
      acd34(12)=dotproduct(ninjaE4,spvae1k2)
      acd34(13)=dotproduct(ninjaE4,spvae1l5)
      acd34(14)=abb34(26)
      acd34(15)=dotproduct(ninjaE4,spvae1e2)
      acd34(16)=abb34(23)
      acd34(17)=dotproduct(ninjaE3,spvae2e1)
      acd34(18)=dotproduct(ninjaE4,spvae1l3)
      acd34(19)=dotproduct(ninjaE4,spvae1l4)
      acd34(20)=abb34(38)
      acd34(21)=dotproduct(ninjaE3,spvae1l4)
      acd34(22)=abb34(52)
      acd34(23)=dotproduct(ninjaE3,spval3e1)
      acd34(24)=abb34(19)
      acd34(25)=abb34(24)
      acd34(26)=dotproduct(ninjaE3,spvae1l5)
      acd34(27)=dotproduct(ninjaE3,spvae1e2)
      acd34(28)=dotproduct(ninjaE3,spval5e1)
      acd34(29)=dotproduct(ninjaA,spvae1k2)
      acd34(30)=dotproduct(ninjaA,spvae1l3)
      acd34(31)=dotproduct(ninjaA,spvak2e1)
      acd34(32)=dotproduct(ninjaA,spvae2e1)
      acd34(33)=dotproduct(ninjaA,spvae1l4)
      acd34(34)=dotproduct(ninjaA,spval3e1)
      acd34(35)=dotproduct(ninjaA,spvae1l5)
      acd34(36)=dotproduct(ninjaA,spvae1e2)
      acd34(37)=dotproduct(ninjaA,spval5e1)
      acd34(38)=abb34(10)
      acd34(39)=abb34(12)
      acd34(40)=abb34(15)
      acd34(41)=abb34(14)
      acd34(42)=abb34(16)
      acd34(43)=abb34(18)
      acd34(44)=abb34(30)
      acd34(45)=abb34(22)
      acd34(46)=abb34(31)
      acd34(47)=abb34(11)
      acd34(48)=acd34(16)*acd34(15)
      acd34(49)=acd34(14)*acd34(13)
      acd34(50)=acd34(3)*acd34(12)
      acd34(48)=acd34(50)+acd34(48)+acd34(49)
      acd34(48)=acd34(48)*acd34(11)
      acd34(49)=acd34(25)*acd34(15)
      acd34(50)=acd34(24)*acd34(13)
      acd34(51)=acd34(5)*acd34(12)
      acd34(49)=acd34(51)+acd34(49)+acd34(50)
      acd34(49)=acd34(49)*acd34(23)
      acd34(50)=acd34(17)*acd34(20)
      acd34(51)=acd34(22)*acd34(28)
      acd34(50)=acd34(50)+acd34(51)
      acd34(50)=acd34(50)*acd34(19)
      acd34(52)=acd34(10)*acd34(9)
      acd34(53)=acd34(8)*acd34(7)
      acd34(52)=acd34(52)+acd34(53)
      acd34(52)=acd34(52)*acd34(6)
      acd34(53)=acd34(5)*acd34(4)
      acd34(54)=acd34(3)*acd34(2)
      acd34(53)=acd34(53)+acd34(54)
      acd34(53)=acd34(53)*acd34(1)
      acd34(54)=acd34(25)*acd34(27)
      acd34(55)=acd34(24)*acd34(26)
      acd34(54)=acd34(54)+acd34(55)
      acd34(55)=acd34(54)*acd34(4)
      acd34(56)=acd34(16)*acd34(27)
      acd34(57)=acd34(14)*acd34(26)
      acd34(56)=acd34(56)+acd34(57)
      acd34(57)=acd34(56)*acd34(2)
      acd34(58)=acd34(10)*acd34(28)
      acd34(59)=acd34(8)*acd34(17)
      acd34(58)=acd34(58)+acd34(59)
      acd34(59)=acd34(58)*acd34(18)
      acd34(60)=acd34(9)*acd34(21)*acd34(22)
      acd34(61)=acd34(20)*acd34(21)
      acd34(62)=acd34(61)*acd34(7)
      acd34(48)=acd34(55)+acd34(57)-acd34(59)-acd34(62)-acd34(60)+acd34(53)-acd&
      &34(50)-acd34(52)+acd34(48)+acd34(49)
      acd34(49)=acd34(25)*acd34(36)
      acd34(50)=acd34(24)*acd34(35)
      acd34(52)=acd34(5)*acd34(29)
      acd34(49)=acd34(50)+acd34(49)+acd34(52)+acd34(43)
      acd34(50)=acd34(23)*acd34(49)
      acd34(52)=-acd34(30)*acd34(58)
      acd34(53)=acd34(34)*acd34(54)
      acd34(54)=acd34(31)*acd34(56)
      acd34(55)=acd34(10)*acd34(37)
      acd34(56)=acd34(8)*acd34(32)
      acd34(55)=acd34(55)+acd34(56)-acd34(39)
      acd34(56)=-acd34(6)*acd34(55)
      acd34(57)=acd34(3)*acd34(31)
      acd34(57)=acd34(57)+acd34(38)
      acd34(58)=acd34(5)*acd34(34)
      acd34(58)=acd34(58)+acd34(57)
      acd34(58)=acd34(1)*acd34(58)
      acd34(59)=acd34(16)*acd34(36)
      acd34(60)=acd34(14)*acd34(35)
      acd34(59)=acd34(40)+acd34(59)+acd34(60)
      acd34(60)=acd34(3)*acd34(29)
      acd34(60)=acd34(60)+acd34(59)
      acd34(60)=acd34(11)*acd34(60)
      acd34(62)=acd34(28)*acd34(46)
      acd34(63)=acd34(27)*acd34(45)
      acd34(64)=acd34(26)*acd34(44)
      acd34(51)=-acd34(33)*acd34(51)
      acd34(65)=acd34(22)*acd34(37)
      acd34(65)=acd34(65)-acd34(42)
      acd34(66)=-acd34(21)*acd34(65)
      acd34(61)=-acd34(32)*acd34(61)
      acd34(67)=-acd34(20)*acd34(33)
      acd34(67)=acd34(41)+acd34(67)
      acd34(67)=acd34(17)*acd34(67)
      acd34(50)=acd34(60)+acd34(50)+acd34(58)+acd34(56)+acd34(67)+acd34(61)+acd&
      &34(66)+acd34(51)+acd34(64)+acd34(62)+acd34(63)+acd34(54)+acd34(53)+acd34&
      &(52)
      acd34(51)=ninjaP*acd34(48)
      acd34(49)=acd34(34)*acd34(49)
      acd34(52)=-acd34(20)*acd34(32)
      acd34(52)=acd34(52)-acd34(65)
      acd34(52)=acd34(33)*acd34(52)
      acd34(53)=-acd34(30)*acd34(55)
      acd34(54)=acd34(31)*acd34(59)
      acd34(55)=acd34(29)*acd34(57)
      acd34(56)=acd34(37)*acd34(46)
      acd34(57)=acd34(36)*acd34(45)
      acd34(58)=acd34(35)*acd34(44)
      acd34(59)=acd34(32)*acd34(41)
      acd34(49)=acd34(51)+acd34(59)+acd34(58)+acd34(57)+acd34(47)+acd34(56)+acd&
      &34(49)+acd34(54)+acd34(53)+acd34(52)+acd34(55)
      brack(ninjaidxt1mu0)=acd34(50)
      brack(ninjaidxt0mu0)=acd34(49)
      brack(ninjaidxt0mu2)=acd34(48)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d34h12_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd34h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d34h12l131_qp

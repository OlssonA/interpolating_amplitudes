module     p2_gg_httbar_d34h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d34h12l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd34h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc34(29)
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l3
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      acc34(1)=abb34(10)
      acc34(2)=abb34(11)
      acc34(3)=abb34(12)
      acc34(4)=abb34(13)
      acc34(5)=abb34(14)
      acc34(6)=abb34(15)
      acc34(7)=abb34(16)
      acc34(8)=abb34(18)
      acc34(9)=abb34(19)
      acc34(10)=abb34(20)
      acc34(11)=abb34(22)
      acc34(12)=abb34(23)
      acc34(13)=abb34(24)
      acc34(14)=abb34(26)
      acc34(15)=abb34(30)
      acc34(16)=abb34(31)
      acc34(17)=abb34(38)
      acc34(18)=abb34(48)
      acc34(19)=abb34(50)
      acc34(20)=abb34(52)
      acc34(21)=acc34(9)*Qspvae1l5
      acc34(22)=acc34(10)*Qspvae1k2
      acc34(23)=acc34(13)*Qspvae1e2
      acc34(21)=acc34(23)+acc34(22)+acc34(21)+acc34(8)
      acc34(21)=Qspval3e1*acc34(21)
      acc34(22)=acc34(4)*Qspvae1k2
      acc34(23)=acc34(12)*Qspvae1e2
      acc34(24)=acc34(14)*Qspvae1l5
      acc34(22)=acc34(24)+acc34(23)+acc34(6)+acc34(22)
      acc34(22)=Qspvak2e1*acc34(22)
      acc34(23)=-acc34(17)*Qspvae2e1
      acc34(24)=-acc34(20)*Qspval5e1
      acc34(23)=acc34(24)+acc34(23)+acc34(7)
      acc34(23)=Qspvae1l4*acc34(23)
      acc34(24)=-acc34(18)*Qspval5e1
      acc34(25)=-acc34(19)*Qspvae2e1
      acc34(24)=acc34(25)+acc34(24)+acc34(3)
      acc34(24)=Qspvae1l3*acc34(24)
      acc34(25)=acc34(1)*Qspvae1k2
      acc34(26)=acc34(5)*Qspvae2e1
      acc34(27)=acc34(11)*Qspvae1e2
      acc34(28)=acc34(15)*Qspvae1l5
      acc34(29)=acc34(16)*Qspval5e1
      brack=acc34(2)+acc34(21)+acc34(22)+acc34(23)+acc34(24)+acc34(25)+acc34(26&
      &)+acc34(27)+acc34(28)+acc34(29)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d34h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd34h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d34
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d34 = 0.0_ki
      d34 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d34, ki), aimag(d34), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d34h12l1_qp

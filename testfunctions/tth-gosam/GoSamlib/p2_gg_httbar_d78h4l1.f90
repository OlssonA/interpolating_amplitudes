module     p2_gg_httbar_d78h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d78h4l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd78h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc78(37)
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3e1
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      QspQ = dotproduct(Q,Q)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      acc78(1)=abb78(9)
      acc78(2)=abb78(10)
      acc78(3)=abb78(11)
      acc78(4)=abb78(12)
      acc78(5)=abb78(13)
      acc78(6)=abb78(14)
      acc78(7)=abb78(15)
      acc78(8)=abb78(16)
      acc78(9)=abb78(17)
      acc78(10)=abb78(18)
      acc78(11)=abb78(19)
      acc78(12)=abb78(20)
      acc78(13)=abb78(21)
      acc78(14)=abb78(22)
      acc78(15)=abb78(23)
      acc78(16)=abb78(24)
      acc78(17)=abb78(25)
      acc78(18)=abb78(27)
      acc78(19)=abb78(29)
      acc78(20)=abb78(31)
      acc78(21)=abb78(34)
      acc78(22)=abb78(36)
      acc78(23)=abb78(37)
      acc78(24)=abb78(38)
      acc78(25)=abb78(39)
      acc78(26)=abb78(40)
      acc78(27)=abb78(41)
      acc78(28)=abb78(42)
      acc78(29)=acc78(7)*Qspvak2e1
      acc78(30)=acc78(18)*Qspval5e1
      acc78(31)=acc78(19)*Qspvae2e1
      acc78(32)=acc78(22)*Qspvae1e2
      acc78(33)=-acc78(26)*Qspvae1l4
      acc78(34)=acc78(27)*Qspvae1l5
      acc78(29)=acc78(34)+acc78(33)+acc78(32)+acc78(31)+acc78(30)+acc78(11)+acc&
      &78(29)
      acc78(29)=QspQ*acc78(29)
      acc78(30)=-acc78(6)*Qspvae1l4
      acc78(31)=acc78(10)*Qspvae1l3
      acc78(32)=acc78(14)*Qspvae1e2
      acc78(33)=acc78(16)*Qspvae1l5
      acc78(34)=Qspvae1k2*acc78(13)
      acc78(30)=acc78(34)+acc78(33)+acc78(32)+acc78(31)+acc78(30)+acc78(5)
      acc78(30)=Qspvak2e1*acc78(30)
      acc78(31)=acc78(3)*Qspvae1l4
      acc78(32)=-acc78(25)*Qspvae1e2
      acc78(33)=-acc78(28)*Qspvae1l5
      acc78(31)=acc78(33)+acc78(32)+acc78(9)+acc78(31)
      acc78(31)=Qspval3e1*acc78(31)
      acc78(32)=acc78(12)*Qspvae2e1
      acc78(33)=acc78(20)*Qspval5e1
      acc78(32)=acc78(33)+acc78(32)+acc78(1)
      acc78(32)=Qspvae1l4*acc78(32)
      acc78(33)=acc78(23)*Qspvae2e1
      acc78(34)=acc78(24)*Qspval5e1
      acc78(33)=acc78(34)+acc78(33)+acc78(4)
      acc78(33)=Qspvae1l3*acc78(33)
      acc78(34)=acc78(8)*Qspvae1e2
      acc78(35)=acc78(15)*Qspval5e1
      acc78(36)=acc78(17)*Qspvae2e1
      acc78(37)=acc78(21)*Qspvae1l5
      brack=acc78(2)+acc78(29)+acc78(30)+acc78(31)+acc78(32)+acc78(33)+acc78(34&
      &)+acc78(35)+acc78(36)+acc78(37)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d78h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd78h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d78
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d78 = 0.0_ki
      d78 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d78, ki), aimag(d78), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d78h4l1

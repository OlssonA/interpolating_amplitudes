module     p2_gg_httbar_d142h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d142h0l1.f90
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
      use p2_gg_httbar_abbrevd142h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc142(40)
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: QspQ
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      QspQ = dotproduct(Q,Q)
      acc142(1)=abb142(12)
      acc142(2)=abb142(13)
      acc142(3)=abb142(14)
      acc142(4)=abb142(15)
      acc142(5)=abb142(16)
      acc142(6)=abb142(17)
      acc142(7)=abb142(18)
      acc142(8)=abb142(19)
      acc142(9)=abb142(20)
      acc142(10)=abb142(21)
      acc142(11)=abb142(22)
      acc142(12)=abb142(26)
      acc142(13)=abb142(28)
      acc142(14)=abb142(29)
      acc142(15)=abb142(33)
      acc142(16)=abb142(36)
      acc142(17)=abb142(41)
      acc142(18)=abb142(45)
      acc142(19)=abb142(74)
      acc142(20)=abb142(76)
      acc142(21)=abb142(77)
      acc142(22)=abb142(79)
      acc142(23)=abb142(82)
      acc142(24)=abb142(90)
      acc142(25)=abb142(96)
      acc142(26)=abb142(98)
      acc142(27)=abb142(102)
      acc142(28)=acc142(5)*Qspvak2e2
      acc142(29)=acc142(7)*Qspvae1e2
      acc142(30)=acc142(10)*Qspvak1e2
      acc142(31)=acc142(14)*Qspval4e2
      acc142(32)=acc142(15)*Qspval5e2
      acc142(28)=acc142(32)+acc142(31)+acc142(30)+acc142(29)+acc142(28)+acc142(&
      &1)
      acc142(28)=Qspvae2k2*acc142(28)
      acc142(29)=-Qspvae2l5*acc142(26)
      acc142(30)=acc142(8)*Qspvae2k1
      acc142(31)=acc142(25)*Qspvae2e1
      acc142(32)=acc142(27)*Qspvae2l4
      acc142(29)=acc142(32)+acc142(31)+acc142(24)+acc142(30)+acc142(29)
      acc142(29)=Qspval5e2*acc142(29)
      acc142(30)=-Qspvae2l4*acc142(26)
      acc142(31)=acc142(6)*Qspvae2k1
      acc142(32)=acc142(20)*Qspvae2e1
      acc142(33)=acc142(22)*Qspvae2l5
      acc142(30)=acc142(33)+acc142(32)+acc142(12)+acc142(31)+acc142(30)
      acc142(30)=Qspval4e2*acc142(30)
      acc142(31)=acc142(21)*Qspvae2e1
      acc142(31)=acc142(31)+acc142(18)
      acc142(31)=Qspvae1e2*acc142(31)
      acc142(32)=-acc142(26)*Qspvae2k1
      acc142(32)=acc142(11)+acc142(32)
      acc142(32)=Qspvak1e2*acc142(32)
      acc142(33)=acc142(2)*Qspvak2e2
      acc142(34)=acc142(4)*Qspvae2k1
      acc142(35)=acc142(17)*Qspvae2e1
      acc142(36)=acc142(19)*Qspvae2l4
      acc142(37)=acc142(23)*Qspvae2l5
      acc142(38)=Qspvae2l3*acc142(13)
      acc142(39)=Qspval3e2*acc142(9)
      acc142(40)=QspQ*acc142(3)
      brack=acc142(16)+acc142(28)+acc142(29)+acc142(30)+acc142(31)+acc142(32)+a&
      &cc142(33)+acc142(34)+acc142(35)+acc142(36)+acc142(37)+acc142(38)+acc142(&
      &39)+acc142(40)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d142h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd142h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d142
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d142 = 0.0_ki
      d142 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d142, ki), aimag(d142), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d142h0l1
